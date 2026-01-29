module SimpleImageRegistration

using NNlib: DenseConvDims, output_size, channels_out
using NNlibCUDA: conv!
using CUDA:
    CuArray,
    CuVector,
    CuMatrix,
    @cuda,
    blockIdx,
    blockDim,
    threadIdx,
    CuContext,
    CuStream,
    @context!,
    CuDefaultStream,
    stream,
    context,
    synchronize
using CUDA.CUFFT:
    cufftHandle,
    cufftPlanMany,
    CUFFT_R2C,
    CUFFT_C2R,
    cufftDestroy,
    cufftExecR2C,
    cufftExecC2R,
    cufftSetStream
using CUDA.CUBLAS: gemm_batched!

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

function CUDA.unsafe_free!(plan::SimpleBatchedCuFFTPlan, stream::CuStream = stream())
    @context! skip_destroyed=true plan.ctx cufftDestroy(plan)
end

unsafe_finalize!(plan::SimpleBatchedCuFFTPlan) = unsafe_free!(plan, CuDefaultStream())

@inline function update_stream(plan::SimpleBatchedCuFFTPlan)
    new_stream = stream()
    if plan.stream != new_stream
        plan.stream = new_stream
        cufftSetStream(plan, new_stream)
    end
    return
end

function make_buffer_maybe_mmap(::Type{T}, sz, scratch_dir = "") where {T}
    if isempty(scratch_dir)
        b = T(undef, sz)
    else
        b = mtemp(scratch_dir) do path, io
            mmap(path, T, sz)
        end
    end
    b
end

struct GpuBatchConvState{T<:DenseConvDims}
    cdims::T
    conv_kern_d::CuArray{Float32,4}

    function GpuBatchConvState(
        conv_kern::AbstractMatrix{Float32},
        img_size::NTuple{2},
        batch_l::Int,
        conv_padding::Integer = div(size(conv_kern, 1), 4),
    )
        csz = size(conv_kern)
        csz[1] == csz[2] || throw(ArgumentError("square conv only"))
        img_conv_size = (img_size..., 1, batch_l)
        ckern_conv_size = (csz..., 1, 1)
        cdims = DenseConvDims(img_conv_size, ckern_conv_size, padding = conv_padding)
        conv_kern_d = CuArray(reshape(conv_kern, ckern_conv_size...))
        new{typeof(cdims)}(cdims, conv_kern_d)
    end
end

_do_conv!(dst, src, s::GpuBatchConvState) = conv!(dst, src, s.conv_kern_d, s.cdims)

mutable struct AlignmentBatchState{T<:DenseConvDims}
    frames_h::Array{Float32,3}
    offsets_h::Matrix{Int32}
    offsets_interp_h::Matrix{Float32}
    n_filled_frames::Int

    frames_d::CuArray{Float32,4}
    padded_frames_d::CuArray{Float32,3}
    npad::Int

    convd_d::CuArray{Float32,4}

    gpu_conv_state::GpuBatchConvState{T}
    img_kern_d::CuArray{Float32,2}

    forward_plan::SimpleBatchedCuFFTPlan
    rev_plan::SimpleBatchedCuFFTPlan
    forward_padded_plan::SimpleBatchedCuFFTPlan
    rev_padded_plan::SimpleBatchedCuFFTPlan

    convd_f_d::CuArray{ComplexF32,3}
    corr_f_d::CuArray{ComplexF32,3}
    corr_f_d_bak::CuArray{ComplexF32,3}

    subsample_use_for_template::Bool
    subsample_use_for_alignment::Bool
    subsample_return_offsets::Bool

    subsample_search_radius::Float32
    subsample_ratio::Int
    subsample_intermediate::CuArray{ComplexF32,3}
    subsample_idft::CuArray{ComplexF32,3}
    subsample_complex_result::CuArray{ComplexF32,3}
    subsample_bdft::CuArray{ComplexF32,3}
    subsample_real_result::CuArray{Float32,3}

    template_f_d::CuArray{ComplexF32,2}
    template_f_d_scratch::CuArray{ComplexF32,2}
    template_count::Int

    padded_frames_f_d::CuArray{ComplexF32,3}

    offsets_d::CuMatrix{Int32}
    offsets_interp_d::CuMatrix{Float32}

    # This holds all of the arrays and plans on host and GPU to do alignment
    # If `scratch_dir` is empty, then the host buffers will be pinned to RAM.
    # Otherwise, they will be memory-mapped files. `npad` controls how many zero
    # pixels will be added to images before they are aligned.
    function AlignmentBatchState(
        nx::Integer,
        ny::Integer,
        batch_l::Integer,
        conv_kern::AbstractMatrix{Float32},
        img_kern::AbstractMatrix{Float32},
        scratch_dir::AbstractString = "";
        npad = 20,
        subsample_search_radius = 1.5,
        subsample_ratio = 4,
        subsample_use_for_template = false,
        subsample_use_for_alignment = true,
        subsample_return_offsets = subsample_use_for_template ||
                                   subsample_use_for_alignment,
        conv_args = (),
    )
        conv_state = GpuBatchConvState(conv_kern, (nx, ny), batch_l, conv_args...)

        b = make_buffer_maybe_mmap(Array{Float32,3}, (nx, ny, batch_l), scratch_dir)
        frames_h = isempty(scratch_dir) ? pin(b) : b
        offsets_h = pin(Matrix{Int32}(undef, 2, batch_l))

        frames_d = CuArray{Float32}(undef, nx, ny, 1, batch_l)
        pad_sz = (nx + 2 * npad, ny + 2 * npad, batch_l)
        padded_frames_d = CuArray{Float32,3}(undef, pad_sz)

        osz = output_size(conv_state.cdims, conv_args...)
        i_win_sz = size(img_kern)
        i_win_sz == osz || throw(ArgumentError("window size wrong"))
        convd_d = CuArray{Float32,4}(undef, osz..., channels_out(conv_state.cdims), batch_l)
        img_kern_d = CuArray(img_kern)


        osz_r = Cint[osz[2], osz[1]]
        forward_plan_ref = Ref{cufftHandle}()
        cufftPlanMany(
            forward_plan_ref,
            2,
            osz_r,
            C_NULL,
            1,
            1,
            C_NULL,
            1,
            1,
            CUFFT_R2C,
            batch_l,
        )
        forward_plan = SimpleBatchedCuFFTPlan(forward_plan_ref[])

        rev_plan_ref = Ref{cufftHandle}()
        cufftPlanMany(
            rev_plan_ref,
            2,
            osz_r,
            C_NULL,
            1,
            1,
            C_NULL,
            1,
            1,
            CUFFT_C2R,
            batch_l,
        )
        rev_plan = SimpleBatchedCuFFTPlan(rev_plan_ref[])

        r_pad_sz = Cint[pad_sz[2], pad_sz[1]]
        forward_padded_plan_ref = Ref{cufftHandle}()
        cufftPlanMany(
            forward_padded_plan_ref,
            2,
            r_pad_sz,
            C_NULL,
            1,
            1,
            C_NULL,
            1,
            1,
            CUFFT_R2C,
            batch_l,
        )
        forward_padded_plan = SimpleBatchedCuFFTPlan(forward_padded_plan_ref[])

        rev_padded_plan_ref = Ref{cufftHandle}()
        cufftPlanMany(
            rev_padded_plan_ref,
            2,
            r_pad_sz,
            C_NULL,
            1,
            1,
            C_NULL,
            1,
            1,
            CUFFT_C2R,
            batch_l,
        )
        rev_padded_plan = SimpleBatchedCuFFTPlan(rev_padded_plan_ref[])

        o_f_sz = (div(osz[1], 2) + 1, osz[2])
        convd_f_d = CuArray{ComplexF32,3}(undef, o_f_sz[1], o_f_sz[2], batch_l)
        corr_f_d = similar(convd_f_d)
        corr_f_d_bak = similar(corr_f_d)

        if subsample_ratio > 1 && (
            subsample_use_for_alignment ||
            subsample_use_for_template ||
            subsample_return_offsets
        )
            ninterp = n_interp_points(subsample_search_radius, subsample_ratio)
            effective_subsample_ratio = subsample_ratio
            offsets_interp_h = pin(similar(offsets_h, Float32))
        else
            ninterp = 0
            effective_subsample_ratio = 1
            offsets_interp_h = similar(offsets_h, Float32, (2, 0))
        end

        subsample_intermediate = CuArray{ComplexF32,3}(undef, o_f_sz[1], ninterp, batch_l)
        subsample_idft = CuArray{ComplexF32,3}(undef, o_f_sz[2], ninterp, batch_l)
        subsample_complex_result = CuArray{ComplexF32,3}(undef, ninterp, ninterp, batch_l)
        subsample_real_result = CuArray{Float32,3}(undef, ninterp, ninterp, batch_l)
        subsample_bdft = CuArray{ComplexF32,3}(undef, ninterp, o_f_sz[1], batch_l)

        offsets_d = CuMatrix{Int32}(undef, 2, batch_l)
        interp_offset_l = ifelse(effective_subsample_ratio > 1, batch_l, 0)
        offsets_interp_d = CuMatrix{Float32}(undef, 2, interp_offset_l)
        template_f_d = CuMatrix{ComplexF32}(undef, o_f_sz)
        template_f_d_scratch = similar(template_f_d)

        pad_f_sz = (div(pad_sz[1], 2) + 1, pad_sz[2])
        padded_frames_f_d = CuArray{ComplexF32,3}(undef, pad_f_sz..., batch_l)
        synchronize()

        new{typeof(conv_state.cdims)}(
            frames_h,
            offsets_h,
            offsets_interp_h,
            0,
            frames_d,
            padded_frames_d,
            npad,
            convd_d,
            conv_state,
            img_kern_d,
            forward_plan,
            rev_plan,
            forward_padded_plan,
            rev_padded_plan,
            convd_f_d,
            corr_f_d,
            corr_f_d_bak,
            subsample_use_for_template,
            subsample_use_for_alignment,
            subsample_return_offsets,
            subsample_search_radius,
            effective_subsample_ratio,
            subsample_intermediate,
            subsample_idft,
            subsample_complex_result,
            subsample_bdft,
            subsample_real_result,
            template_f_d,
            template_f_d_scratch,
            -1,
            padded_frames_f_d,
            offsets_d,
            offsets_interp_d,
        )
    end
end

@inline device_to_fft_freq_unnorm(i, n, v = div(n, 2)) = CUDA.mod(i + v, n) - v

function transfer_shift_ndxs!(offsets, max_ndxs, n)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        @inbounds offsets[1, i] = max_ndxs[i][1] - 1
        @inbounds offsets[2, i] = max_ndxs[i][2] - 1
    end
    nothing
end

function undo_shift_transformed!(transformed, offsets, nx, nxfull, ny, nz)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    z = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if x <= nx && y <= ny && z <= nz
        i = (z - 1) * ny * nx + (y - 1) * nx + x
        @inbounds x_exp = -offsets[1, z] * device_to_fft_freq_unnorm(x, nxfull) / nxfull
        @inbounds y_exp = -offsets[2, z] * device_to_fft_freq_unnorm(y, ny) / ny
        @inbounds transformed[i] = transformed[i] * CUDA.exp(-2 * pi * im * (x_exp + y_exp))
    end
    nothing
end

function fill_idft!(
    idftarr,
    offsets,
    nx,
    ny,
    nz,
    ntrans,
    nside,
    interp_ratio,
    n_half,
    upper_offset,
)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    z = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    interp_offset = (y - 1 - nside) / interp_ratio
    if x <= nx && y <= ny && z <= nz
        interp_pos = interp_offset + offsets[2, z]
        freq_pos = x - 1 + ifelse(x > n_half, upper_offset, 0)
        idftarr[x, y, z] = conj_unity_root(freq_pos * interp_pos, ntrans)
    end
    nothing
end

function fill_bdft!(
    bdftarr,
    offsets,
    nx,
    ny,
    nz,
    ntrans,
    nside,
    interp_ratio,
    last_doubled_freq,
)
    x = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    y = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    z = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    interp_offset = (x - 1 - nside) / interp_ratio
    if x <= nx && y <= ny && z <= nz
        coeff = CUDA.ifelse(1 < y <= last_doubled_freq, 2, 1)
        interp_pos = interp_offset + offsets[1, z]
        bdftarr[x, y, z] = coeff * conj_unity_root((y - 1) * interp_pos, ntrans)
    end
    nothing
end

function determine_absolute_subsampled_max!(
    interp_offsets,
    whole_offsets,
    relmaxpos,
    n,
    nside,
    interp_ratio,
    nx,
    ny,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= n
        unwrapped_x = whole_offsets[1, i] + (relmaxpos[i][1] - 1 - nside) / interp_ratio
        interp_offsets[1, i] = device_to_fft_freq_unnorm(unwrapped_x, nx)
        unwrapped_y = whole_offsets[2, i] + (relmaxpos[i][2] - 1 - nside) / interp_ratio
        interp_offsets[2, i] = device_to_fft_freq_unnorm(unwrapped_y, ny)
    end
    nothing
end

function initialize_template!(s::AlignmentBatchState)
    # Convolve frames
    _do_conv!(s.convd_d, s.frames_d, s.gpu_conv_state)

    # Find mean windowed frame
    nxc, nyc, _, batch_l = size(s.convd_d)
    k = s.n_filled_frames
    if k < batch_l
        s.convd_d[:, :, 1, (k+1):batch_l] .= 0.0f0
    end
    mean_frame_d = CuMatrix{Float32}(undef, nxc, nyc)
    fill!(mean_frame_d, 0.0f0)
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
    r = (offset+1):min(offset+batch_l, nf)
    kin = length(r)
    r, kin
end

function wrap_offsets!(offsets_d, nx, ny)
    @sync begin
        @async begin
            CUDA.@sync offsets_d[1, :] .= device_to_fft_freq_unnorm.(offsets_d[1, :], nx)
            nothing
        end
        @async begin
            CUDA.@sync offsets_d[2, :] .= device_to_fft_freq_unnorm.(offsets_d[2, :], ny)
            nothing
        end
    end
end

function subsample_align!(s::AlignmentBatchState, shift_block_size)
    nx, ny, batch_l = size(s.frames_h)
    nxc, nyc, _, _ = size(s.convd_d)
    o_f_sz = size(s.convd_f_d)[1:2]
    ninterp = size(s.subsample_real_result, 1)
    nside = div(ninterp - 1, 2)
    nx_upsampled, ny_upsampled = floor.(Int, (nxc, nyc) .* s.subsample_ratio)
    n_half = div(nyc - 1, 2) + 1
    upper_offset = ny_upsampled - nyc
    subsample_idft_view = [view(s.subsample_idft,:,:,i) for i = 1:batch_l]
    subsample_bdft_view = [view(s.subsample_bdft,:,:,i) for i = 1:batch_l]
    subsample_complex_res_view = [view(s.subsample_complex_result,:,:,i) for i = 1:batch_l]

    subsample_intermediate_view = [view(s.subsample_intermediate,:,:,i) for i = 1:batch_l]
    corr_mat_view = [view(s.corr_f_d_bak,:,:,i) for i = 1:batch_l]
    @sync begin
        @async CUDA.@sync begin
            idft_mat_grid_size = cld.(size(s.subsample_idft), shift_block_size)
            @cuda(
                blocks=idft_mat_grid_size,
                threads = shift_block_size,
                fill_idft!(
                    s.subsample_idft,
                    s.offsets_d,
                    o_f_sz[2],
                    ninterp,
                    batch_l,
                    o_f_sz[2],
                    nside,
                    s.subsample_ratio,
                    n_half,
                    upper_offset,
                )
            )
            fill!(s.subsample_intermediate, 0)
            gemm_batched!(
                'N',
                'N',
                1,
                corr_mat_view,
                subsample_idft_view,
                0,
                subsample_intermediate_view,
            )
            nothing
        end
        @async CUDA.@sync begin
            last_doubled_freq = 1 + div(nx_upsampled - 1, 2)
            bdft_mat_grid_size = cld.(size(s.subsample_bdft), shift_block_size)
            @cuda(
                blocks=bdft_mat_grid_size,
                threads = shift_block_size,
                fill_bdft!(
                    s.subsample_bdft,
                    s.offsets_d,
                    ninterp,
                    o_f_sz[1],
                    batch_l,
                    nxc,
                    nside,
                    s.subsample_ratio,
                    last_doubled_freq,
                )
            )
            nothing
        end
    end
    CUDA.@sync begin
        fill!(s.subsample_complex_result, 0)
        gemm_batched!(
            'N',
            'N',
            1,
            subsample_bdft_view,
            subsample_intermediate_view,
            0,
            subsample_complex_res_view,
        )
        s.subsample_real_result .= real.(s.subsample_complex_result)
        _, maxpos_rel_d = findmax(
            s.subsample_real_result,
            dims = (1, 2),
        )::Tuple{CuArray{Float32,3},CuArray{CartesianIndex{3},3}}
        nblocks_transfer = cld(batch_l, 256)
        @cuda(
            blocks=nblocks_transfer,
            threads=256,
            determine_absolute_subsampled_max!(
                s.offsets_interp_d,
                s.offsets_d,
                maxpos_rel_d,
                batch_l,
                nside,
                s.subsample_ratio,
                nxc,
                nyc,
            )
        )
    end
    wrap_offsets!(s.offsets_d, nxc, nyc)
end

function process_batch!(s::AlignmentBatchState)
    nx, ny, batch_l = size(s.frames_h)
    nxc, nyc, _, _ = size(s.convd_d)
    frames_packed_d = dropdims(s.frames_d, dims = 3)
    o_f_sz = size(s.convd_f_d)[1:2]
    npad = s.npad
    shift_block_size = (16, 16, 1)
    @sync begin
        @async CUDA.@sync begin
            # Convolve frames
            _do_conv!(s.convd_d, s.frames_d, s.gpu_conv_state)

            # Apply window function to convolved result
            s.convd_d .*= s.img_kern_d

            # Fourier transform frames
            convd_packed_d = dropdims(s.convd_d, dims = 3)
            update_stream(s.forward_plan)
            cufftExecR2C(s.forward_plan, s.convd_d, s.convd_f_d)

            # Correlate frames with template in frequency domain
            s.corr_f_d .= CUDA.conj.(s.template_f_d) .* s.convd_f_d
            copyto!(s.corr_f_d_bak, s.corr_f_d)
            # Inverse Fourier transform of correlation
            update_stream(s.rev_plan)
            cufftExecC2R(s.rev_plan, s.corr_f_d, s.convd_d)

            # Find maximum correlation (locate best shift)
            _, maxpos_d = findmax(
                convd_packed_d,
                dims = (1, 2),
            )::Tuple{CuArray{Float32,3},CuArray{CartesianIndex{3},3}}
            nblocks_transfer = cld(batch_l, 256)
            @cuda(
                blocks=nblocks_transfer,
                threads=256,
                transfer_shift_ndxs!(s.offsets_d, maxpos_d, batch_l)
            )
            nothing
        end
        @async CUDA.@sync begin
            # Pad frames
            fill!(s.padded_frames_d, 0)
            s.padded_frames_d[(1+npad):(npad+nx), (1+npad):(npad+ny), :] .= frames_packed_d
            # Fourier transform padded frames
            update_stream(s.forward_padded_plan)
            cufftExecR2C(s.forward_padded_plan, s.padded_frames_d, s.padded_frames_f_d)
            nothing
        end
    end
    if s.subsample_ratio > 1
        subsample_align!(s, shift_block_size)
    else
        wrap_offsets!(s.offsets_d, nxc, nyc)
    end
    @sync begin
        k = s.n_filled_frames
        @async CUDA.@sync begin
            # Shift convolved frames and update template
            new_count = s.template_count + k
            @sync begin
                @async CUDA.@sync begin
                    nxcf = div(nxc, 2) + 1
                    tmpl_shift_grid_size = cld.((nxcf, nyc, batch_l), shift_block_size)
                    offsets_to_use =
                        s.subsample_use_for_template ? s.offsets_interp_d : s.offsets_d
                    @cuda(
                        blocks=tmpl_shift_grid_size,
                        threads=shift_block_size,
                        undo_shift_transformed!(
                            s.convd_f_d,
                            offsets_to_use,
                            nxcf,
                            nxc,
                            nyc,
                            batch_l,
                        )
                    )
                    if k < batch_l
                        s.convd_f_d[:, :, (k+1):batch_l] .= 0
                    end
                    nothing
                end
                @async begin
                    CUDA.@sync s.template_f_d .*= (s.template_count / new_count)
                    nothing
                end
                @async begin
                    CUDA.@sync fill!(s.template_f_d_scratch, 0)
                    nothing
                end
            end
            mapreducedim!(identity, +, s.template_f_d_scratch, s.convd_f_d)
            s.template_f_d .+= s.template_f_d_scratch .* (k / new_count)
            s.template_count = new_count
            nothing
        end

        # Align padded frames
        @async CUDA.@sync begin
            nxp, nyp, _ = size(s.padded_frames_d)
            nxpf = div(nxp, 2) + 1
            algn_shift_grid_size = cld.((nxpf, nyp, batch_l), shift_block_size)
            offsets_to_use =
                s.subsample_use_for_alignment ? s.offsets_interp_d : s.offsets_d
            @cuda(
                blocks=algn_shift_grid_size,
                threads=shift_block_size,
                undo_shift_transformed!(
                    s.padded_frames_f_d,
                    offsets_to_use,
                    nxpf,
                    nxp,
                    nyp,
                    batch_l,
                )
            )

            update_stream(s.rev_padded_plan)
            cufftExecC2R(s.rev_padded_plan, s.padded_frames_f_d, s.padded_frames_d)
            s.padded_frames_d ./= nxp * nyp
            frames_packed_d .= s.padded_frames_d[(1+npad):(npad+nx), (1+npad):(npad+ny), :]
            copyto!(s.frames_h, frames_packed_d)
            nothing
        end

        @async begin
            copyto!(s.offsets_h, s.offsets_d)
            nothing
        end
        if s.subsample_ratio > 1
            @async begin
                copyto!(s.offsets_interp_h, s.offsets_interp_d)
                nothing
            end
        end
    end
    s.frames_h, s.offsets_h, s.offsets_interp_h, s.n_filled_frames
end

function collect_frames!(framesarr, frames::AbstractArray{<:Any,3}, r)
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
function run_alignment!(
    sink_f::F,
    s::AlignmentBatchState,
    frames::AbstractArray{<:Any,3},
    scratch_dir = "";
    init_l = nothing,
) where {F<:Function}
    # buffer input data, transfer to GPU, run alignment, buffer outputs and
    # pass to sink_f
    nf = size(frames, 3)
    nx, ny, batch_l = size(s.frames_h)
    nbatch = cld(nf, batch_l)
    nbatch < 1 && return
    bin = make_buffer_maybe_mmap(Array{Float32,3}, (nx, ny, batch_l), scratch_dir)
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
        bout = make_buffer_maybe_mmap(Array{Float32,3}, (nx, ny, batch_l), scratch_dir)
        boffsets = Matrix{Float32}(undef, 2, batch_l)
        if s.subsample_ratio > 1
            boffsets_interp = similar(boffsets)
        else
            boffsets_interp = similar(boffsets, (2, 0))
        end
        r, kin = batch_range(2, batch_l, nf)
        t = @sync begin
            @async bin[:, :, 1:kin] .= frames[:, :, r]
            t = @async process_batch!(s)
            t
        end
        aligned, offsets, offsets_interp, k = fetch(t)
        # output buffers now contain results of first batch
        @sync begin
            @async copyto!(boffsets, offsets)
            @async copyto!(boffsets_interp, offsets_interp)
            @async copyto!(bout, aligned)
        end
        for batch_no = 3:nbatch
            # Aligning batch_no - 1 batch, preparing data for batch_no batch,
            # handling output of batch_no - 2 batch
            transfer_frames!(s, bin, 1:kin)
            r, kin = batch_range(batch_no, batch_l, nf)
            t = @sync begin
                @async sink_f(bout, boffsets, boffsets_interp, k)
                @async bin[:, :, 1:kin] .= frames[:, :, r]
                t = @async process_batch!(s)
                t
            end

            _, _, _, k = fetch(t)
            @sync begin
                @async copyto!(boffsets, offsets)
                @async copyto!(boffsets_interp, offsets_interp)
                @async copyto!(bout, aligned)
            end
        end
        transfer_frames!(s, bin, 1:kin)
        # handling output of second to last batch, aligning last batch
        t = @sync begin
            @async sink_f(bout, boffsets, boffsets_interp, k)
            t = @async process_batch!(s)
            t
        end
        # Handling output of last batch
        _, _, _, k = fetch(t)
        sink_f(aligned, offsets, offsets_interp, k)
    else
        sink_f(process_batch!(s)...)
    end
end

# Method for using already created GPU state
function align_video_data!(
    outfile,
    s::AlignmentBatchState,
    vdata::AbstractArray{<:Any,3},
    scratch_dir = "";
    kwargs...,
)
    nx, ny, nf = size(vdata)
    offsets = Matrix{Float32}(undef, 2, nf)
    if s.subsample_ratio > 1
        offsets_interp = similar(offsets)
    else
        offsets_interp = similar(offsets, (2, 0))
    end
    outpos = 0
    open(outfile, "w") do iow
        run_alignment!(
            s,
            vdata,
            scratch_dir;
            kwargs...,
        ) do aligned_frames, alignment_offsets, align_offsets_interp, k
            write(iow, view(aligned_frames,:,:,(1:k)))
            out_range = (outpos+1):(outpos+k)
            offsets[:, out_range] .= alignment_offsets[:, 1:k]
            if !isempty(align_offsets_interp)
                offsets_interp[:, out_range] .= align_offsets_interp[:, 1:k]
            end
            outpos += k
        end
    end
    offsets, offsets_interp, outpos
end

# This is the high-level function that writes the aligned data to disk
function align_video_data(
    outfile,
    vdata::AbstractArray{<:Any,3},
    conv_kern,
    img_kern,
    scratch_dir = "";
    batch_l = 200,
    kwargs...,
)
    nx, ny, nf = size(vdata)
    s = AlignmentBatchState(nx, ny, batch_l, conv_kern, img_kern, scratch_dir; kwargs...)
    align_video_data!(outfile, s, vdata; kwargs...)
end

conj_unity_root(k, n) = CUDA.exp(im * 2 * pi * CUDA.mod(k, n) / n)
unity_root(k, n) = CUDA.exp(-im * 2 * pi * CUDA.mod(k, n) / n)

n_interp_side_points(r, s) = max(floor(Int, r * s), 0)
n_interp_points(r, s) = 2 * n_interp_side_points(r, s) + 1
function interp_grid(c, r, s)
    nside = n_interp_side_points(r, s)
    c .+ ((-nside):nside) ./ s
end

function _convolve_data!(convd_out, img_in, conv_state, batch_l)
    nx, ny, nf = size(img_in)
    nxc, nyc = output_size(conv_state.cdims)
    ninbuff = 2
    inbuffs_empty = Channel{CuArray{Float32,4}}(ninbuff)
    outbuffs_empty = Channel{CuArray{Float32,4}}(ninbuff)
    for i = 1:ninbuff
        put!(inbuffs_empty, CuArray{Float32,4}(undef, (nx, ny, 1, batch_l)))
        put!(outbuffs_empty, CuArray{Float32,4}(undef, (nxc, nyc, 1, batch_l)))
    end

    inbuffs_filled = Channel{Tuple{CuArray{Float32,4},UnitRange{Int}}}(1)
    outbuffs_filled = Channel{Tuple{CuArray{Float32,4},UnitRange{Int}}}(1)
    frames_h = pin(Array{Float32,4}(undef, nx, ny, 1, batch_l))
    frames_convd_h = pin(Array{Float32,4}(undef, nxc, nyc, 1, batch_l))
    @sync begin
        feeder_t = @async begin
            nbatch = cld(nf, batch_l)
            for i = 1:nbatch
                r, k = batch_range(i, batch_l, nf)
                frames_h[:, :, 1, 1:k] .= img_in[:, :, r]
                buff_d = take!(inbuffs_empty)
                copyto!(buff_d, frames_h)
                put!(inbuffs_filled, (buff_d, r))
            end
        end
        conv_t = @async begin
            for (buff_d, r) in inbuffs_filled
                convd_d = take!(outbuffs_empty)
                CUDA.@sync _do_conv!(convd_d, buff_d, conv_state)
                put!(inbuffs_empty, buff_d)
                put!(outbuffs_filled, (convd_d, r))
            end
        end
        sink_t = @async begin
            for (convd_d, r) in outbuffs_filled
                copyto!(frames_convd_h, convd_d)
                put!(outbuffs_empty, convd_d)
                convd_out[:, :, r] .= frames_convd_h[:, :, 1, 1:length(r)]
            end
        end
        bind(inbuffs_filled, feeder_t)
        bind(outbuffs_filled, conv_t)
    end
    close(inbuffs_empty)
    close(outbuffs_empty)
end

end
