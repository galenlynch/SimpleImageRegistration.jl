module SimpleImageRegistration

using NNlib: DenseConvDims, output_size, channels_out
using NNlibCUDA: cudnnConvolutionForward
using CUDA: CuArray, CuVector, CuMatrix, @cuda, blockIdx, blockDim, threadIdx,
    CuContext, CuStream, @context!, CuDefaultStream, stream, context,
    synchronize
using CUDA.CUFFT: cufftHandle, cufftPlanMany, CUFFT_R2C, CUFFT_C2R,
    cufftDestroy, cufftExecR2C, cufftExecC2R, cufftSetStream

using CUDA.Mem: pin

import CUDA
import CUDA.CUFFT
using Mmap: mmap

using LinearAlgebra: mul!

using Base: mapreducedim!, RefValue

import Base: unsafe_convert, convert

import CUDA: unsafe_free!, unsafe_finalize!

mutable struct SimpleBatchedCuFFTPlan
    handle::cufftHandle
    ctx::CuContext
    stream::CuStream

    function SimpleBatchedCuFFTPlan(handle::cufftHandle, stream::CuStream = stream())
        p = new(handle, context(), stream)
        finalizer(unsafe_finalize!, p)
        p
    end
end

Base.unsafe_convert(::Type{cufftHandle}, p::SimpleBatchedCuFFTPlan) = p.handle
Base.convert(::Type{cufftHandle}, p::SimpleBatchedCuFFTPlan) = p.handle

function CUDA.unsafe_free!(plan::SimpleBatchedCuFFTPlan,
                           stream::CuStream=stream())
    @context! skip_destroyed=true plan.ctx cufftDestroy(plan)
end

unsafe_finalize!(plan::SimpleBatchedCuFFTPlan) =
    unsafe_free!(plan, CuDefaultStream())

@inline function update_stream(plan::SimpleBatchedCuFFTPlan)
    new_stream = stream()
    if plan.stream != new_stream
        plan.stream = new_stream
        cufftSetStream(plan, new_stream)
    end
    return
end

function make_buffer_maybe_mmap(::Type{T}, sz, scratch_dir = "") where T
    if isempty(scratch_dir)
        b = T(undef, sz)
    else
        b = mtemp(scratch_dir) do path, io
            mmap(path, T, sz)
        end
    end
    b
end

mutable struct AlignmentBatchState{T<:DenseConvDims}
    frames_h::Array{Float32, 3}
    offsets_h::Matrix{Float32}
    n_filled_frames::Int

    frames_d::CuArray{Float32, 4}
    padded_frames_d::CuArray{Float32, 3}
    npad::Int

    convd_d::CuArray{Float32, 4}

    cdims::T
    conv_kern_d::CuArray{Float32, 4}
    img_kern_d::CuArray{Float32, 2}

    forward_plan::SimpleBatchedCuFFTPlan
    rev_plan::SimpleBatchedCuFFTPlan
    forward_padded_plan::SimpleBatchedCuFFTPlan
    rev_padded_plan::SimpleBatchedCuFFTPlan

    convd_f_d::CuArray{Complex{Float32}, 3}
    corr_f_d::CuArray{Complex{Float32}, 3}
    template_f_d::CuArray{Complex{Float32}, 2}
    template_f_d_scratch::CuArray{Complex{Float32}, 2}
    template_count::Int

    padded_frames_f_d::CuArray{Complex{Float32}, 3}

    offsets_d::CuMatrix{Float32}

    # This holds all of the arrays and plans on host and GPU to do alignment
    # If `scratch_dir` is empty, then the host buffers will be pinned to RAM,
    # otherwise, they will be memory-mapped files. `npad` controls how many
    # zero pixels images will be imbedded in before they are aligned.
    # `use_phase_correlation` controls whether alignment is done through
    # correlation or phase-correlation. I've only tested correlation so
    # caveat emptor.
    function AlignmentBatchState(nx::Integer, ny::Integer, batch_l::Integer,
                                 conv_kern::AbstractMatrix{Float32},
                                 img_kern::AbstractMatrix{Float32},
                                 scratch_dir::AbstractString = "";
                                 npad = 20)
        csz = size(conv_kern)
        csz[1] == csz[2] || throw(ArgumentError("square conv only"))
        ckern_l = csz[1]
        conv_kern_d = CuArray(reshape(conv_kern, size(conv_kern)..., 1, 1))
        b = make_buffer_maybe_mmap(Array{Float32, 3}, (nx, ny, batch_l),
                                   scratch_dir)
        frames_h = isempty(scratch_dir) ? pin(b) : b
        offsets_h = pin(Matrix{Float32}(undef, 2, batch_l))

        frames_d = CuArray{Float32}(undef, nx, ny, 1, batch_l)
        pad_sz = (nx + 2 * npad, ny + 2 * npad, batch_l)
        padded_frames_d = CuArray{Float32, 3}(undef, pad_sz)
        cdims = DenseConvDims(frames_d, conv_kern_d, padding = div(csz[1], 4))
        osz = output_size(cdims)
        i_win_sz = size(img_kern)
        i_win_sz == osz || throw(ArgumentError("window size wrong"))
        convd_d = CuArray{Float32, 4}(undef, osz..., channels_out(cdims),
                                      batch_l)
        img_kern_d = CuArray(img_kern)


        rosz = Cint[osz[2], osz[1]]
        forward_plan_ref = Ref{cufftHandle}()
        cufftPlanMany(forward_plan_ref, 2, rosz, C_NULL, 1, 1, C_NULL, 1, 1,
                      CUFFT_R2C, batch_l)
        forward_plan = SimpleBatchedCuFFTPlan(forward_plan_ref[])

        rev_plan_ref = Ref{cufftHandle}()
        cufftPlanMany(rev_plan_ref, 2, rosz, C_NULL, 1, 1, C_NULL, 1, 1,
                      CUFFT_C2R, batch_l)
        rev_plan = SimpleBatchedCuFFTPlan(rev_plan_ref[])

        r_pad_sz = Cint[pad_sz[2], pad_sz[1]]
        forward_padded_plan_ref = Ref{cufftHandle}()
        cufftPlanMany(forward_padded_plan_ref, 2, r_pad_sz, C_NULL, 1, 1,
                      C_NULL, 1, 1, CUFFT_R2C, batch_l)
        forward_padded_plan = SimpleBatchedCuFFTPlan(forward_padded_plan_ref[])

        rev_padded_plan_ref = Ref{cufftHandle}()
        cufftPlanMany(rev_padded_plan_ref, 2, r_pad_sz, C_NULL, 1, 1, C_NULL, 1,
                      1, CUFFT_C2R, batch_l)
        rev_padded_plan = SimpleBatchedCuFFTPlan(rev_padded_plan_ref[])

        o_f_sz = (div(osz[1], 2) + 1, osz[2])
        convd_f_d = CuArray{Complex{Float32}, 3}(undef, o_f_sz[1], o_f_sz[2],
                                                 batch_l)
        corr_f_d = similar(convd_f_d)
        offsets_d = CuMatrix{Float32}(undef, 2, batch_l)
        template_f_d = CuMatrix{Complex{Float32}}(undef, o_f_sz)
        template_f_d_scratch = similar(template_f_d)

        pad_f_sz = (div(pad_sz[1], 2) + 1, pad_sz[2])
        padded_frames_f_d = CuArray{Complex{Float32}, 3}(undef, pad_f_sz...,
                                                         batch_l)
        synchronize()

        new{typeof(cdims)}(frames_h, offsets_h, 0, frames_d, padded_frames_d,
                           npad, convd_d, cdims, conv_kern_d, img_kern_d,
                           forward_plan, rev_plan, forward_padded_plan,
                           rev_padded_plan, convd_f_d, corr_f_d, template_f_d,
                           template_f_d_scratch, -1, padded_frames_f_d,
                           offsets_d)
    end
end

@inline device_to_fft_freq_unnorm(i, n, v = div(n, 2)) =
    CUDA.mod(i - 1 + v, n) - v

function transfer_shift_ndxs!(offsets, max_ndxs, n, nx, ny)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds offsets[1, i] = device_to_fft_freq_unnorm(max_ndxs[i][1], nx)
        @inbounds offsets[2, i] = device_to_fft_freq_unnorm(max_ndxs[i][2], ny)
    end
    nothing
end

function undo_shift_transformed!(transformed, offsets, nx, nxfull, ny, nz)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    z = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if x <= nx && y <= ny && z <= nz
        i = (z - 1) * ny * nx + (y - 1) * nx + x
        @inbounds x_exp = -offsets[1, z] *
            device_to_fft_freq_unnorm(x, nxfull) / nxfull
        @inbounds y_exp = -offsets[2, z] *
            device_to_fft_freq_unnorm(y, ny) / ny
        @inbounds transformed[i] = transformed[i] *
            CUDA.exp(-2 * pi * im * (x_exp + y_exp))
    end
    nothing
end

function initialize_template!(s::AlignmentBatchState)
    # Convolve frames
    cudnnConvolutionForward(s.convd_d, s.frames_d, s.conv_kern_d, s.cdims)

    # Find mean windowed frame
    nxc, nyc, _, batch_l = size(s.convd_d)
    k = s.n_filled_frames
    if k < batch_l
        s.convd_d[:, :, 1, k+1:batch_l] .= 0f0
    end
    mean_frame_d = CuMatrix{Float32}(undef, nxc, nyc)
    fill!(mean_frame_d, 0f0)
    mapreducedim!(identity, +, mean_frame_d, s.convd_d)
    # Apply window
    mean_frame_d .= mean_frame_d .* s.img_kern_d ./ k

    # Fourier transform mean frame
    init_forward_plan = CUFFT.plan_rfft(mean_frame_d)
    mul!(s.template_f_d, init_forward_plan, mean_frame_d)
    s.template_count = 0
    synchronize()
end

function batch_range(batch_no, batch_l, nf)
    offset = (batch_no - 1) * batch_l
    r = offset + 1 : min(offset + batch_l, nf)
    kin = length(r)
    r, kin
end

function process_batch!(s::AlignmentBatchState)
    nx, ny, batch_l = size(s.frames_h)
    nxc, nyc, _, _ = size(s.convd_d)
    frames_packed_d = dropdims(s.frames_d, dims = 3)
    npad = s.npad
    @sync begin
        @async CUDA.@sync begin
            # Convolve frames
            cudnnConvolutionForward(s.convd_d, s.frames_d, s.conv_kern_d,
                                    s.cdims)

            # Apply window function to convolved result
            s.convd_d .*= s.img_kern_d

            # Fourier transform frames
            convd_packed_d = dropdims(s.convd_d, dims = 3)
            update_stream(s.forward_plan)
            cufftExecR2C(s.forward_plan, convd_packed_d, s.convd_f_d)

            # Correlate frames with template in frequency domain
            s.corr_f_d .= CUDA.conj.(s.template_f_d) .* s.convd_f_d
            # Inverse Fourier transform of correlation
            update_stream(s.rev_plan)
            cufftExecC2R(s.rev_plan, s.corr_f_d, convd_packed_d)

            # Find maximum correlation (locate best shift)
            maxvals_d, maxpos_d =
                findmax(convd_packed_d, dims = (1, 2))::Tuple{
                    CuArray{Float32, 3}, CuArray{CartesianIndex{3}, 3}
                }
            nblocks_transfer = cld(batch_l, 256)
            @cuda(blocks=nblocks_transfer, threads=256,
                  transfer_shift_ndxs!(s.offsets_d, maxpos_d, batch_l, nxc, nyc))
        end
        @async CUDA.@sync begin
            # Pad frames
            fill!(s.padded_frames_d, 0)
            s.padded_frames_d[1 + npad : npad + nx, 1 + npad : npad + ny, :] .=
                frames_packed_d
            # Fourier transform padded frames
            update_stream(s.forward_padded_plan)
            cufftExecR2C(s.forward_padded_plan, s.padded_frames_d,
                         s.padded_frames_f_d)
        end
    end
    shift_block_size = (16, 16, 1)
    @sync begin
        k = s.n_filled_frames
        @async CUDA.@sync begin
            # Shift convolved frames and update template
            new_count = s.template_count + k
            @sync begin
                @async CUDA.@sync begin
                    nxcf = div(nxc, 2) + 1
                    tmpl_shift_grid_size = cld.((nxcf, nyc, batch_l),
                                                shift_block_size)
                    @cuda(blocks=tmpl_shift_grid_size, threads=shift_block_size,
                          undo_shift_transformed!(s.convd_f_d, s.offsets_d, nxcf,
                                             nxc, nyc, batch_l))
                    if k < batch_l
                        s.convd_f_d[:, :, k + 1 : batch_l] .= 0
                    end
                end
                @async CUDA.@sync s.template_f_d .*=
                    (s.template_count / new_count)
                @async CUDA.@sync fill!(s.template_f_d_scratch, 0)
            end
            mapreducedim!(identity, +, s.template_f_d_scratch, s.convd_f_d)
            s.template_f_d .+= s.template_f_d_scratch .* (k / new_count)
            s.template_count = new_count
        end

        # Align padded frames
        @async CUDA.@sync begin
            nxp, nyp, _ = size(s.padded_frames_d)
            nxpf = div(nxp, 2) + 1
            algn_shift_grid_size = cld.((nxpf, nyp, batch_l), shift_block_size)
            @cuda(blocks=algn_shift_grid_size, threads=shift_block_size,
                  undo_shift_transformed!(s.padded_frames_f_d, s.offsets_d, nxpf,
                                     nxp, nyp, batch_l))

            update_stream(s.rev_padded_plan)
            cufftExecC2R(s.rev_padded_plan, s.padded_frames_f_d,
                         s.padded_frames_d)
            s.padded_frames_d ./= nxp * nyp
            frames_packed_d .=
                s.padded_frames_d[1 + npad : npad + nx, 1 + npad : npad + ny, :]
            copyto!(s.frames_h, frames_packed_d)
        end

        @async copyto!(s.offsets_h, s.offsets_d)
    end
    s.frames_h, s.offsets_h, s.n_filled_frames
end

function collect_frames!(framesarr, frames::AbstractArray{<:Any, 3}, r)
    k = length(r)
    framesarr[:, :, 1:k] .= frames[:, :, r]
end

function collect_frames!(framesarr, frames, r)
    for (i, fno) in enumerate(r)
        framesarr[:, :, i] .= frames(fno)
    end
    framesarr
end

function transfer_frames!(s::AlignmentBatchState, frames, r)
    k = length(r)
    nx, ny, batch_l = size(s.frames_h)
    k <= batch_l || throw(ArgumentError("range too large"))
    collect_frames!(s.frames_h, frames, r)
    copyto!(s.frames_d, reshape(s.frames_h, nx, ny, 1, batch_l))
    s.n_filled_frames = k
    synchronize()
end

# This version "triple buffers" and is not the most memory efficient
function run_alignment!(sink_f::F, s::AlignmentBatchState,
                        frames::AbstractArray{<:Any, 3}, scratch_dir = "";
                        init_l = nothing) where F <: Function
    # buffer input data, transfer to GPU, run alignment, buffer outputs and
    # pass to sink_f
    nf = size(frames, 3)
    nx, ny, batch_l = size(s.frames_h)
    nbatch = cld(nf, batch_l)
    nbatch < 1 && return
    bin = make_buffer_maybe_mmap(Array{Float32, 3}, (nx, ny, batch_l),
                                 scratch_dir)
    # Find initial template
    truncated_init = init_l !== nothing && init_l < min(nf, batch_l)
    ninit_raw = min(batch_l, nf)
    ninit = truncated_init ? min(ninit_raw, init_l) : ninit_raw
    if truncated_init
        transfer_frames!(s, frames, 1:ninit)
        @sync begin
            @async initialize_template!(s)
            @async bin[:, :, 1:ninit_raw] .= frames[:, :, 1:ninit_raw]
        end
        transfer_frames!(s, bin, 1:ninit_raw)
    else
        transfer_frames!(s, frames, 1:ninit)
        initialize_template!(s)
    end
    # Do alignment
    if nbatch > 1
        # aligning first batch, preparing data for second batch
        bout = make_buffer_maybe_mmap(Array{Float32, 3},
                                      (nx, ny, batch_l), scratch_dir)
        boffsets = Matrix{Float32}(undef, 2, batch_l)
        r, kin = batch_range(2, batch_l, nf)
        t = @sync begin
            @async bin[:, :, 1:kin] .= frames[:, :, r]
            t = @async process_batch!(s)
            t
        end
        aligned, offsets, k = fetch(t)
        # output buffers now contain results of first batch
        copyto!(boffsets, offsets)
        copyto!(bout, aligned)
        for batch_no in 3:nbatch
            # Aligning batch_no - 1 batch, preparing data for batch_no batch,
            # handling output of batch_no - 2 batch
            transfer_frames!(s, bin, 1:kin)
            r, kin = batch_range(batch_no, batch_l, nf)
            t = @sync begin
                @async sink_f(bout, boffsets, k)
                @async bin[:, :, 1:kin] .= frames[:, :, r]
                t = @async process_batch!(s)
                t
            end

            _, _, k = fetch(t)
            copyto!(boffsets, offsets)
            copyto!(bout, aligned)
        end
        transfer_frames!(s, bin, 1:kin)
        # handling output of second to last batch, aligning last batch
        t = @sync begin
            @async sink_f(bout, boffsets, k)
            t = @async process_batch!(s)
            t
        end
        # Handling output of last batch
         _, _, k = fetch(t)
        sink_f(aligned, offsets, k)
    else
        sink_f(process_batch!(s)...)
    end
end

# Method for using already created GPU state
function align_video_data!(outfile, s::AlignmentBatchState,
                           vdata::AbstractArray{<:Any, 3}, scratch_dir = "";
                           kwargs...)
    nx, ny, nf = size(vdata)
    offsets = Matrix{Float32}(undef, 2, nf)
    outpos = 0
    open(outfile, "w") do iow
        run_alignment!(s, vdata, scratch_dir; kwargs...
            ) do aligned_frames, alignment_offsets, k
                write(iow, view(aligned_frames, :, :, 1:k))
                out_range = outpos + 1: outpos + k
                offsets[:, out_range] .= alignment_offsets[:, 1:k]
                outpos += k
            end
    end
    offsets, outpos
end

# This is the high-level function that writes the aligned data to disk
function align_video_data(outfile, vdata::AbstractArray{<:Any, 3}, conv_kern,
                          img_kern, scratch_dir = ""; batch_l = 200, kwargs...)
    nx, ny, nf = size(vdata)
    s = AlignmentBatchState(nx, ny, batch_l, conv_kern, img_kern, scratch_dir;
                            kwargs...)
    align_video_data!(outfile, s, vdata; kwargs...)
end

end
