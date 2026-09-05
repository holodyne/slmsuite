from slmsuite.holography.algorithms._header import *
from slmsuite.holography.algorithms._hologram import Hologram


# amp * exp(1j * phase) * phasor, in one pass, avoiding three full (S, h, w)
# temporaries per transform. `amp`/`phase` are broadcast across the S planes by
# the caller.
_nearfield_build = None
if cp is not np:
    _nearfield_build = cp.ElementwiseKernel(
        "T amp, T phase, C phasor",
        "C nearfield",
        "nearfield = phasor * C(amp * cos(phase), amp * sin(phase));",
        "multiplane_nearfield_build",
    )


class MultiplaneHologram(Hologram):
    """
    Holography combining multiple objectives, potentially across planes of focus or color.
    Other :class:`Hologram` subclasses are restricted to either optimizing a hologram
    within a fixed basis of spots or
    within the grid of a discrete Fourier transform at a fixed plane of focus.
    This :class:`MultiplaneHologram` acts as a metaclass to optimize many individual
    holograms simultaneously---over many planes or pointsets---producing a composite
    phase pattern.

    Note
    ~~~~
    Though the infrastructure to make this trivial is not yet in place,
    the idea of a 'plane' extends to planes of color. That is, this class
    :class:`MultiplaneHologram` could also be used to optimize a multicolor hologram and
    account for how the farfield of each color scales with wavelength.

    Tip
    ~~~
    Calls to :meth:`.optimize()` which update :attr:`flags` also update the flags of any
    child hologram.

    Attributes
    ----------
    holograms : list of :class:`Hologram`
        List of sub-holograms to optimize simultaneously.
    weights : list of float
        Weight for each hologram. This allows the user to redistribute power between
        holograms. Keep in mind that each hologram will normalize itself, so differences
        in intensity between target patterns cannot be relied upon.
    """

    def __init__(self, holograms, weights=None):
        """
        Initializes a 'meta' hologram consisting of several sub-holograms optimizing at
        the same time.

        Parameters
        ----------
        holograms : list of :class:`Hologram`
            List of ``N`` sub-holograms to optimize simultaneously.
        weights : array_like of float OR None
            List of ``N`` floats.
            If ``None``, defaults to even power.
        """
        self.holograms = holograms

        # Check that all holograms are actually holograms and not MultiplaneHolograms.
        for h in self.holograms:
            if "MultiplaneHologram" in str(type(h)):
                raise ValueError("Multiplane hologram recursion is not supported.")
            if "Hologram" not in str(type(h)):
                raise ValueError(
                    f"Multiplane hologram must be provided child holograms, not {type(h)}"
                )

        # Construct the parent hologram with empty goals but complete context.
        super().__init__(
            target=holograms[0].slm_shape,  # This hologram has a fake target.
            amp=holograms[0].amp,
            phase=holograms[0].phase,
            slm_shape=holograms[0].slm_shape,
            dtype=holograms[0].dtype,
        )
        self.target = None

        # Force all the child holograms to point to the same data.
        for h in self.holograms:
            h.amp = self.amp
            h.phase = self.phase

        # Parse weights
        if weights is None:
            weights = np.ones(len(self), dtype=self.dtype)

        self.weights = np.array(
            weights,
            copy=(False if np.__version__[0] == "1" else None),
            dtype=self.dtype,
        )
        self.weights /= Hologram._norm(self.weights, xp=np)

        # Batched-FFT fast path, when the children share shape, dtype, and transforms.
        self._batched = self._can_batch()
        if self._batched:
            self._batched_setup()

    def _can_batch(self):
        """Whether the children can share a single batched FFT2 call."""
        if cp is np:
            return False
        if len(self.holograms) < 2:
            return False
        h0 = self.holograms[0]
        for h in self.holograms[1:]:
            if tuple(h.shape) != tuple(h0.shape):
                return False
            if tuple(h.slm_shape) != tuple(h0.slm_shape):
                return False
            if h.dtype != h0.dtype:
                return False
        # Only batch subclasses that use Hologram's default transforms.
        # Subclasses with custom transforms (e.g. CompressedSpotHologram)
        # override these methods.
        for h in self.holograms:
            if type(h)._nearfield2farfield is not Hologram._nearfield2farfield:
                return False
            if type(h)._farfield2nearfield is not Hologram._farfield2nearfield:
                return False
        return True

    def _can_batch_routines(self, mraf_variables):
        """
        Whether :meth:`_gs_farfield_routines` can run as a single pass over the
        batched tensors instead of looping the children. Requires that every
        child would take the same branch: the plain (non-MRAF) path, with no
        per-child weighting or phase-fixing decisions. Flags are shared with
        the children by :meth:`_update_flags`.
        """
        if not self._batched:
            return False
        if "WGS" in self.flags.get("method", ""):
            return False
        if self.flags.get("fixed_phase", False):
            return False
        return not any(m["mraf_enabled"] for m in mraf_variables)

    def _batched_setup(self):
        """Allocate the batched scratch tensors used by the FFT fast path."""
        S = len(self.holograms)
        h0 = self.holograms[0]
        shape = tuple(h0.shape)
        complex_dtype = h0.dtype_complex
        real_dtype = h0.dtype

        self._batched_nearfield = cp.zeros((S,) + shape, dtype=complex_dtype)
        self._batched_amp_ff = cp.zeros((S,) + shape, dtype=real_dtype)
        self._batched_phase_ff = cp.zeros((S,) + shape, dtype=real_dtype)
        # _batched_farfield gets reassigned by every cp.fft.fft2 call; init it
        # so children can be rebound before the first FFT (e.g. for tests).
        self._batched_farfield = cp.zeros((S,) + shape, dtype=complex_dtype)

        # Stacked child propagation_kernels and their phasor, and the stacked
        # child target-amplitude weights used by the batched amplitude
        # replacement. Both are refreshed through _restack_children; see there
        # for the invalidation scheme.
        self._batched_kernels = cp.zeros((S,) + tuple(h0.slm_shape), dtype=real_dtype)
        self._batched_kernel_ids = [None] * S
        self._batched_kernel_phasor = None  # (S, slm_h, slm_w) complex

        self._batched_child_weights = cp.zeros((S,) + shape, dtype=real_dtype)
        self._batched_child_weights_ids = [None] * S

        # Cache the meta weights on GPU; refreshed if self.weights changes.
        self._batched_weights_cp = None
        self._batched_weights_id = None

    def _restack_children(self, attr, buffer, ids):
        """
        Refresh ``buffer[i]`` from each child's ``attr`` if any child rebound
        it since the last call, and report whether anything changed.

        Detected via ``id()``, which is sound here because each array is kept
        alive by the child that owns it: methods like :meth:`reset_weights` and
        :meth:`set_propagation_kernels` rebind the attribute to a fresh array
        rather than writing into the existing one, so an identity check catches
        exactly the updates that matter. ``None`` leaves zeros in place.

        Caution
        ~~~~~~~
        This detects *rebinding*, not in-place mutation: writing into a child's
        existing ``propagation_kernel`` or ``weights`` array leaves the stacked
        copy stale. Use :meth:`set_propagation_kernels` to update kernels.
        Content-hashing the arrays instead would catch that case, but costs a
        device synchronization per child per transform -- measured at ~36% of a
        GS iteration -- so it is deliberately not done.
        """
        if all(id(getattr(h, attr)) == cached for h, cached in
               zip(self.holograms, ids)):
            return False

        for i, h in enumerate(self.holograms):
            value = getattr(h, attr)
            if value is None:
                buffer[i] = 0
            else:
                # Broadcasting handles both scalars and full arrays.
                buffer[i] = value
            ids[i] = id(value)
        return True

    def set_propagation_kernels(self, kernels):
        """
        Set every child's :attr:`propagation_kernel` at once from a stacked
        array.

        The batched fast path works from a single ``(S, slm_h, slm_w)`` tensor
        internally, so assigning the children one at a time forces it to
        restack them -- an allocation, ``S`` scatter-assignments, and a dtype
        cast -- every time. Handing the stack over directly skips all of that.
        Each child's :attr:`propagation_kernel` is rebound to the corresponding
        slice, so the per-child and non-batched code paths keep reading what
        they did before.

        Parameters
        ----------
        kernels : array_like
            Phase kernels of shape ``(S, slm_h, slm_w)``, or anything
            broadcastable to it (e.g. a single ``(slm_h, slm_w)`` array applied
            to every plane).
        """
        h0 = self.holograms[0]
        target_shape = (len(self.holograms),) + tuple(h0.slm_shape)

        kernels = cp.ascontiguousarray(
            cp.broadcast_to(cp.asarray(kernels, dtype=h0.dtype), target_shape)
        )

        for i, h in enumerate(self.holograms):
            h.propagation_kernel = kernels[i]

        if self._batched:
            self._batched_kernels = kernels
            self._batched_kernel_ids = [id(h.propagation_kernel) for h in
                                        self.holograms]
            self._rebuild_batched_kernel_phasor()

    def _rebuild_batched_kernel_phasor(self):
        """
        Cache ``exp(1j * kernel)``. Used by both :meth:`_nearfield2farfield_batched` and
        :meth:`_farfield2nearfield_batched` (via ``cp.conj`` on the same tensor).
        """
        self._batched_kernel_phasor = cp.exp(
            1j * self._batched_kernels.astype(self.holograms[0].dtype_complex)
        )

    def _refresh_batched_kernels(self):
        """Rebuild the stacked phasor if any child's propagation_kernel was reassigned."""
        # The `is None` arm covers the case where the phasor was never built:
        # children that all start with propagation_kernel=None leave the stack
        # matching its initial state, so a restack alone would not be
        # triggered.
        if (self._restack_children("propagation_kernel", self._batched_kernels,
                                   self._batched_kernel_ids) or
            self._batched_kernel_phasor is None):
            self._rebuild_batched_kernel_phasor()

    def _refresh_batched_child_weights(self):
        """Restack the children's target-amplitude ``weights`` if any was
        reassigned."""
        self._restack_children( "weights", self._batched_child_weights,
                               self._batched_child_weights_ids)

    def _refresh_batched_weights(self):
        """Mirror self.weights (numpy) onto the GPU."""
        if self._batched_weights_id != id(self.weights):
            self._batched_weights_cp = cp.asarray(
                self.weights, dtype=self.holograms[0].dtype_complex
            )
            self._batched_weights_id = id(self.weights)

    def __len__(self):
        return len(self.holograms)

    def __getitem__(self, index):
        return self.holograms[index]

    @staticmethod
    def get_multiplane_defocus_blur(
        cameraslm,
        targets,
        target_depths,
        return_depths=None,
        sharp_focus=True,
    ):
        """
        From a stack of target (power) images at ``target_depths``, generate a stack
        of images at ``return_depths``, accounting for defocus blur.
        Power is summed as if all depths were transparent; i.e. objects do not block
        objects further behind.
        This is a partial farfield implementation of
        `realistic defocus blur <https://doi.org/10.48550/arXiv.2205.07030>`_.

        Warning
        -------
        This feature seems to lead to less stable holography without, perhaps, some
        additional optimizations.

        Parameters
        ----------
        cameraslm : ~slmsuite.hardware.cameraslms.FourierSLM
            Hardware to implement blur for. Calibrations are necessary to determine how
            much to blur.

            Tip
            ---
            Right now, the blurring is Gaussian and analytic, but in the future, the
            measurement of the point spread function should be used.
        targets : array_like
            Stack of images of shape ``(image_count, h, w)``.
        target_depths : list of float
            Depths in ``"kxy"`` focal power units corresponding to the ``targets``.
        return_depths : list of float
            Depths to return images at.
            If ``None``, use ``target_depths``.
        sharp_focus : bool
            If ``False``, depths at focal planes are blurred by the point spread radius
            of a focused spot.
            If ``True``, all the blurring is reduced by the focused point spread radius,
            keeping images that are in focus sharp.
        """
        # Parse return_depths.
        if return_depths is None:
            return_depths = target_depths

        # Parse targets.
        if len(np.shape(targets)) != 3:
            raise ValueError("Expected 3D stack of 2D images.")

        (image_count, h, w) = np.shape(targets)

        # Check target_depths.
        if image_count != len(target_depths):
            raise ValueError(
                "There should be the same number of images as target_depths."
            )

        # Make the return data and gather useful parameters.
        canvas = np.zeros((len(return_depths), h, w))

        # Gather f_eff in cam_pix/rad.
        if cameraslm.cam.pitch_um is None:
            raise ValueError(
                "Camera pitch_um is necessary to calculate defocus blur. "
                "Otherwise, we have no reference for the scale of a wavelength."
            )

        f_eff = cameraslm.get_effective_focal_length("ij")
        w0_kxy = cameraslm.slm.get_spot_radius_kxy()
        w0_pix = f_eff * w0_kxy
        w0_um = w0_pix * np.mean(cameraslm.cam.pitch_um)

        zr = np.pi * w0_um * w0_um / cameraslm.slm.wav_um  # (what if n != 1?)

        for j, z2 in enumerate(return_depths):
            for i, z1 in enumerate(target_depths):
                dz = (z1 - z2) * (f_eff * f_eff)

                blur = w0_pix * (
                    np.sqrt(1 + (dz / zr) ** 2) - (1 if sharp_focus else 0)
                )
                blur = 2 * int(blur) + 1

                canvas[j, :, :] += cv2.GaussianBlur(targets[i], (blur, blur), 0)

        return canvas

    # Overload user functions with meta functionality.

    @property
    def _feedback_supported(self):
        """Whatever any child can act on; the flags are pushed down to all of them."""
        supported = set().union(*[set(h._feedback_supported) for h in self.holograms])
        return tuple(sorted(supported)) if supported else ("computational",)

    def _update_flags(self, method, feedback, stat_groups, **kwargs):
        # First update the parent flags.
        super()._update_flags(method, feedback, stat_groups, **kwargs)

        # Then update each of the child flags.
        for h in self.holograms:
            h.flags.update(self.flags)

    def _update_weights(self, *args, **kwargs):
        for h in self.holograms:
            h._update_weights(*args, **kwargs)

    def _get_target_moments_knm_norm(self):
        # Get the data from the child holograms.
        centers = []
        stds = []
        for h in self.holograms:
            center, std = h._get_target_moments_knm_norm()
            # A single-spot plane has zero variance (std == 0), which would give a
            # 0/0 in the analytic variance integral below. Force a floor of one
            # pixel of width; std is in normalized-knm units, so one pixel == 1/shape.
            std = np.maximum(std, 1.0 / np.flip(np.asarray(h.shape, dtype=float)))
            centers.append(center)
            stds.append(std)

        # Weight the centers.
        centers = np.vstack(centers)
        center = np.nansum(np.square(self.weights).reshape(-1, 1) * centers, axis=0)

        # With the center, now weight the stds. We're doing an analytic integration of
        # x^2 over rectangles corresponding to the center \pm sqrt(3) * std of each hologram.
        stds = np.vstack(stds)

        c = centers - center.reshape(1, 2)
        l = c - stds * np.sqrt(3)
        r = c + stds * np.sqrt(3)

        integral_normalized = (r * r * r - l * l * l) / (2 * stds * np.sqrt(3)) / 3
        std = np.sqrt(
            np.nansum(
                np.square(self.weights).reshape(-1, 1) * integral_normalized, axis=0
            )
        )

        return center, std

    def reset(self, reset_phase=True, reset_flags=False, reset_weights=True):
        # Resetting the phase of the parent resets the phase of the children because
        # phase is shared.
        super().reset(reset_phase, reset_flags, reset_weights)

        # Reset the other child variables. Weights are excluded: the
        # reset_weights() call inside super().reset() above is this class's
        # override, which already loops every child.
        for h in self.holograms:
            h.reset(reset_phase=False, reset_flags=reset_flags, reset_weights=False)

    def reset_weights(self):
        for h in self.holograms:
            h.reset_weights()

    def plot_farfield(self, *args, **kwargs):
        for h in self.holograms:
            h.plot_farfield(*args, **kwargs)

    # def plot_nearfield(self, *args, **kwargs):
    #     for h in self.holograms: h.plot_nearfield(*args, **kwargs)

    def plot_stats(self, *args, **kwargs):
        for h in self.holograms:
            h.plot_stats(*args, **kwargs)

    def _update_stats(self, stat_groups=None):
        # FUTURE: make meta stat group.
        for h in self.holograms:
            h._update_stats(stat_groups)

    def set_target(self, *args, **kwargs):
        raise RuntimeError(
            "Do not use MultiplaneHologram.set_target(). "
            "Instead, update the targets of the children holograms directly."
        )

    # Multiplane hacks to get meta optimization to work.

    def _cg_loss(self, phase_torch):
        """Sum the losses of all the child holograms."""
        loss = self.holograms[0]._cg_loss(phase_torch)

        for h in self.holograms[1:]:
            loss += h._cg_loss(phase_torch)

        return loss

    def _nearfield2farfield(self, phase_torch=None):
        """Have all the holograms populate their own farfield variables."""
        # phase_torch is accepted for base-class compatibility; meta CG routes through
        # _cg_loss -> child _cg_loss, so it is always None here.
        if self._batched:
            self._nearfield2farfield_batched()
            return
        for h in self.holograms:
            h._nearfield2farfield()
            h.iter = self.iter

    def _farfield2nearfield(self, extract=True):
        """Sum all the complex nearfields together for the meta nearfield."""
        if self._batched:
            self._farfield2nearfield_batched()
            return

        self.nearfield.fill(0)

        for h, w in zip(self.holograms, self.weights):
            h._farfield2nearfield(extract=False)  # Avoid individually extracting phase.

            (i0, i1, i2, i3) = h._unpad_slice

            # Add the complex individual nearfields to our meta nearfield.
            if h.propagation_kernel is None:
                self.nearfield += w * h.nearfield[i0:i1, i2:i3]
            else:
                # Remove the propagation kernel if necessary.
                self.nearfield += (
                    w * h.nearfield[i0:i1, i2:i3] * cp.exp(-1j * h.propagation_kernel)
                )
            h.iter = self.iter

        # Get meta self phase.
        if extract:
            self._nearfield_extract()

    def _invalidate_phase(self):
        """Invalidate every child's cached measurement; the children share the meta phase."""
        for h in self.holograms:
            h._invalidate_phase()

    def _nearfield_extract(self):
        """Extract the meta phase; the children share it, so their measurements go too."""
        super()._nearfield_extract()
        self._invalidate_phase()

    def reset_phase(self, *args, **kwargs):
        """Randomize or set the meta phase."""
        super().reset_phase(*args, **kwargs)
        self._invalidate_phase()

    def _nearfield2farfield_batched(self):
        """Batched FFT2 across child planes. See `_can_batch` for preconditions."""
        self._refresh_batched_kernels()

        h0 = self.holograms[0]
        (i0, i1, i2, i3) = h0._unpad_slice

        # Build the batched nearfield in place: amp * exp(1j * phase),
        # broadcast across the S child planes, times the cached exp(1j *
        # propagation_kernel).
        # The write below covers the whole tensor whenever the hologram is unpadded, so
        # only zero the padding when there is any. Compare against h0.shape (the extent of
        # _batched_nearfield), not self.shape, which is the meta hologram's slm_shape.
        if (i1 - i0, i3 - i2) != tuple(h0.shape):
            self._batched_nearfield.fill(0)

        _nearfield_build(
            self.amp if np.ndim(self.amp) == 0 else self.amp[None, :, :],
            self.phase[None, :, :],
            self._batched_kernel_phasor,
            self._batched_nearfield[:, i0:i1, i2:i3])

        # One batched FFT2 over the trailing axes.
        self._batched_farfield = cp.fft.fftshift(
            cp.fft.fft2(
                cp.fft.fftshift(self._batched_nearfield, axes=(-2, -1)),
                axes=(-2, -1),
                norm="ortho",
            ),
            axes=(-2, -1),
        )
        # Update farfield amplitudes (in-place into the batched scratch tensor).
        cp.abs(self._batched_farfield, out=self._batched_amp_ff)

        # Rebind child views so `_gs_farfield_routines`' in-place writes land
        # in the batched tensors. Re-done every iter because cp.fft.fft2
        # returns a new allocation each call.
        for i, h in enumerate(self.holograms):
            h.farfield = self._batched_farfield[i]
            h.amp_ff = self._batched_amp_ff[i]
            h.phase_ff = self._batched_phase_ff[i]
            h.iter = self.iter

    def _farfield2nearfield_batched(self):
        """Batched IFFT2 + weighted reduction back to the shared nearfield."""
        # One batched IFFT2; reads from self._batched_farfield, which the
        # children may have mutated in `_gs_farfield_routines`.
        self._batched_nearfield = cp.fft.ifftshift(
            cp.fft.ifft2(
                cp.fft.ifftshift(self._batched_farfield, axes=(-2, -1)),
                axes=(-2, -1),
                norm="ortho",
            ),
            axes=(-2, -1),
        )
        # Rebind child views (ifft2 returns a new allocation).
        for i, h in enumerate(self.holograms):
            h.nearfield = self._batched_nearfield[i]
            h.iter = self.iter

        # Weighted reduction:  meta_nearfield = sum_s w_s * nf[s, slice] * conj(phasor[s])
        h0 = self.holograms[0]
        (i0, i1, i2, i3) = h0._unpad_slice
        self._refresh_batched_weights()

        # cupy.einsum does not support out=, so use a tensordot-style reduction.
        # weighted = weights[:, None, None] * nf[:, slice] * conj(phasor) -> (S, slm_h, slm_w)
        # Then sum over axis 0 directly into self.nearfield.
        weighted = (
            self._batched_weights_cp[:, None, None]
            * self._batched_nearfield[:, i0:i1, i2:i3]
            * cp.conj(self._batched_kernel_phasor)
        )
        cp.sum(weighted, axis=0, out=self.nearfield)

        # Extract meta self.phase from self.nearfield.
        self._nearfield_extract()

    def _mraf_helper_routines(self):
        return [h._mraf_helper_routines() for h in self.holograms]

    def _gs_farfield_routines(self, mraf_variables):
        if self._can_batch_routines(mraf_variables):
            self._gs_farfield_routines_batched()
            return
        for h, mraf in zip(self.holograms, mraf_variables):
            h._gs_farfield_routines(mraf)

    def _gs_farfield_routines_batched(self):
        """
        Amplitude replacement for every child in one pass over the batched
        tensors.

        Identical to what the per-child loop does on the plain (non-MRAF) GS
        path -- keep the computed farfield phase, substitute the target
        amplitudes -- but as three elementwise calls on the ``(S, h, w)``
        tensors rather than ``3 * S`` calls on ``(h, w)`` slices. See
        :meth:`_can_batch_routines` for when this applies.
        """
        self._refresh_batched_child_weights()

        cp.arctan2(
            self._batched_farfield.imag,
            self._batched_farfield.real,
            out=self._batched_phase_ff,
        )
        cp.exp(1j * self._batched_phase_ff, out=self._batched_farfield)
        cp.multiply( self._batched_farfield, self._batched_child_weights,
                    out=self._batched_farfield)

    def _remove_vortices(self):
        for h in self.holograms:
            h._remove_vortices()
