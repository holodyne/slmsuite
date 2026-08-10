from slmsuite.holography.algorithms._header import *
from slmsuite._plotting import _slmsuite_plt_show

class _HologramStats(object):

    # Statistics handling.
    @staticmethod
    def _calculate_stats(
        feedback_amp,
        target_amp,
        xp=cp,
        efficiency_compensation=True,
        total=None,
        raw=False
    ):
        """
        Helper function to analyze how close the feedback is to the target.

        Parameters
        ----------
        feedback_amp : numpy.ndarray OR cupy.ndarray
            Computational or measured result of holography.
        target_amp : numpy.ndarray OR cupy.ndarray
            Target of holography.
        xp : module
            This function is used by both :mod:`cupy` and :mod:`numpy`, so we have the option
            to use either. Defaults to :mod:`cupy`.
        efficiency_compensation : bool
            Whether to scale the ``feedback`` based on the overlap with the ``target``.
            This is more accurate for images, but less accurate for SpotHolograms.
        total : float OR None
            Total power recorded by the feedback mechanism. This may differ from the
            power concentrated in ``feedback_amp ** 2`` because, for instance, power
            might exist outside spot integration regions.
            If ``None``, uses an overlap integral method to compute efficiency.
        raw : bool
            Passes the ``"raw_stats"`` flag. If ``True``, stores the
            raw feedback and raw feedback-target ratio for each pixel or spot instead of
            only the derived and compressed statistics.
        """
        # Downgrade to numpy if necessary
        if xp == np and (hasattr(feedback_amp, "get") or hasattr(target_amp, "get")):
            if hasattr(feedback_amp, "get"):
                feedback_amp = feedback_amp.get()

            if hasattr(target_amp, "get"):
                target_amp = target_amp.get()

            if total is not None:
                total = float(total)

        feedback_amp = xp.array(feedback_amp, copy=True)
        target_amp = xp.array(target_amp, copy=True)

        feedback_pwr = xp.square(feedback_amp)
        target_pwr = xp.square(target_amp)

        # If total is provided, calculate efficiency before normalization.
        if total is not None:
            efficiency = xp.nansum(feedback_pwr) / total
            # self._stats_pinned[0]

        # Normalize. Guard against an all-zero feedback/target (e.g. a blank target or a
        # dark first frame), which would otherwise divide by zero and poison every stat.
        feedback_pwr_sum = xp.sum(feedback_pwr)
        if feedback_pwr_sum > 0:
            feedback_pwr *= 1 / feedback_pwr_sum
            feedback_amp *= 1 / xp.sqrt(feedback_pwr_sum)

        target_pwr_sum = xp.nansum(target_pwr)
        if target_pwr_sum > 0:
            target_pwr *= 1 / target_pwr_sum
            target_amp *= 1 / xp.sqrt(target_pwr_sum)

        if total is None:
            # Efficiency overlap integral.
            efficiency_intermediate = xp.nansum(
                xp.multiply(target_amp, feedback_amp)
            )
            efficiency = xp.square(float(efficiency_intermediate))
            if efficiency_compensation and efficiency != 0:
                feedback_pwr *= 1 / efficiency

        # Make some helper lists; ignoring power where target is zero.
        mask = xp.logical_and(target_pwr != 0, xp.logical_not(xp.isnan(target_pwr)))

        feedback_pwr_masked = feedback_pwr[mask]
        target_pwr_masked = target_pwr[mask]

        ratio_pwr = xp.divide(feedback_pwr_masked, target_pwr_masked)
        pwr_err = target_pwr_masked - feedback_pwr_masked

        # Compute the remaining stats. An empty mask (all-zero/all-nan target) leaves the
        # derived stats undefined rather than crashing amin/amax.
        if ratio_pwr.size == 0:
            rmin = rmax = uniformity = np.nan
            pkpk_err = std_err = np.nan
        else:
            rmin = float(xp.amin(ratio_pwr))
            rmax = float(xp.amax(ratio_pwr))
            uniformity = 1 - (rmax - rmin) / (rmax + rmin) if (rmax + rmin) != 0 else np.nan

            pkpk_err = pwr_err.size * float(xp.amax(pwr_err) - xp.amin(pwr_err))
            std_err = pwr_err.size * float(xp.std(pwr_err))

        final_stats = {
            "efficiency": float(efficiency),
            "uniformity": float(uniformity),
            "pkpk_err": float(pkpk_err),
            "std_err": float(std_err),
        }

        if raw:
            # Force a host array, else a device array leaks into the saved stats.
            ratio_pwr_full = np.full(target_pwr.shape, np.nan)

            if xp == np:
                final_stats["raw_pwr"] = np.square(feedback_amp)
                ratio_pwr_full[mask] = ratio_pwr
            else:
                final_stats["raw_pwr"] = xp.square(feedback_amp).get()
                ratio_pwr_full[mask.get()] = ratio_pwr.get()

            final_stats["raw_pwr_ratio"] = ratio_pwr_full

        return final_stats

    def _calculate_stats_computational(self, stats, stat_groups=None):
        """
        Wrapped by :meth:`Hologram._update_stats()`.
        """
        if stat_groups is None: stat_groups = []
        if "computational" in stat_groups:
            stats["computational"] = self._calculate_stats(
                self.amp_ff,
                self.target,
                efficiency_compensation=False,
                raw="raw_stats" in self.flags and self.flags["raw_stats"],
            )

    def _update_stats_dictionary(self, stats):
        """
        Helper function to manage additions to the :attr:`stats`.

        Parameters
        ----------
        stats : dict of dicts
            Dictionary of groups, each group containing a dictionary of stats.
        """
        # Update methods
        M = len(self.stats["method"])
        diff = self.iter + 1 - M
        if diff > 0:  # Extend methods
            self.stats["method"].extend(["" for _ in range(diff)])
            M = self.iter + 1
        self.stats["method"][self.iter] = self.flags["method"]  # Update method

        # Update flags
        flaglist = set(self.flags.keys()).union(set(self.stats["flags"].keys()))
        for flag in flaglist:
            # Extend flag
            if not flag in self.stats["flags"]:
                self.stats["flags"][flag] = [np.nan for _ in range(M)]
            else:
                diff = self.iter + 1 - len(self.stats["flags"][flag])
                if diff > 0:
                    self.stats["flags"][flag].extend([np.nan for _ in range(diff)])

            # Update flag
            if flag in self.flags:
                self.stats["flags"][flag][self.iter] = self.flags[flag]

        # Update stats
        grouplist = set(stats.keys()).union(set(self.stats["stats"].keys()))
        if len(grouplist) > 0:
            statlists = [set(stats[group].keys()) for group in stats.keys()]
            if len(self.stats["stats"].keys()) > 0:
                key = next(iter(self.stats["stats"]))
                statlists.append(set(self.stats["stats"][key].keys()))
            statlist = set.union(*statlists)

            for group in grouplist:
                # Check this group
                if not group in self.stats["stats"]:
                    self.stats["stats"][group] = {}

                if len(statlist) > 0:
                    for stat in statlist:
                        # Extend stat
                        if not stat in self.stats["stats"][group]:
                            self.stats["stats"][group][stat] = [np.nan for _ in range(M)]
                        else:
                            diff = self.iter + 1 - len(self.stats["stats"][group][stat])
                            if diff > 0:
                                self.stats["stats"][group][stat].extend(
                                    [np.nan for _ in range(diff)]
                                )

                        # Update stat
                        if group in stats.keys() and stat in stats[group].keys():
                            self.stats["stats"][group][stat][self.iter] = stats[group][stat]

        # Rawest stats
        if "raw_stats" in self.flags and self.flags["raw_stats"]:
            if not "raw_farfield" in self.stats:
                self.stats["raw_farfield"] = []

            diff = self.iter + 1 - len(self.stats["raw_farfield"])
            if diff > 0:
                self.stats["raw_farfield"].extend(
                    [np.nan for _ in range(diff)]
                )

            if hasattr(self.farfield, "get"):
                farfield = self.farfield.get()
            else:
                farfield = self.farfield.copy()

            self.stats["raw_farfield"][self.iter] = farfield

    def _update_stats(self, stat_groups=None):
        """
        Calculate statistics corresponding to the desired ``stat_groups``.

        Parameters
        ----------
        stat_groups : list of str
            Which groups or types of statistics to analyze.
        """
        stats = {}

        self._calculate_stats_computational(stats, stat_groups)

        self._update_stats_dictionary(stats)

    # These functions are somewhat superseded by pickle.

    def save_stats(self, file_path, include_state=True):
        """
        Uses :meth:`save_h5` to export the statistics hierarchy to a given h5 file.

        Tip
        ~~~
        Enabling the ``"raw_stats"`` flag will save feedback data from each iteration
        instead of only derived statistics. Consider enabling this to save more detailed
        information.

        Parameters
        ----------
        file_path : str
            Full path to the file to write the data to.
        include_state : bool
            If ``True``, also includes all other attributes of :class:`Hologram`
            except for :attr:`dtype` (cannot pickle) and :attr:`amp_ff` (can regenerate).
            These attributes are converted to :mod:`numpy` if necessary.
            Note that the intent is **not** to produce a
            runnable :class:`Hologram` by default (as this would require pickling hardware
            interfaces), but rather to provide extra information for debugging.
        """
        # Save attributes, converting to numpy when necessary.
        to_save = {}

        if include_state:
            to_save_keys = [
                "slm_shape",
                "phase",
                "amp",
                "shape",
                "target",
                "weights",
                "phase_ff",
                "iter",
                "flags",
            ]
            to_save = {}

            for key in to_save_keys:
                value = getattr(self, key)

                if hasattr(value, "get") and not isinstance(value, dict):
                    to_save[key] = value.get()
                else:
                    to_save[key] = value

        # Save stats.
        to_save["stats"] = self.stats

        save_h5(file_path, to_save)

    def load_stats(self, file_path, include_state=True):
        """
        Uses :meth:`load_h5` to import the statistics hierarchy from a given h5 file.


        Parameters
        ----------
        file_path : str
            Full path to the file to read the data from.
        include_state : bool
            If ``True``, also overwrite all other attributes of :class:`Hologram`
            except for :attr:`dtype` and :attr:`amp_ff`.
        """
        from_save = load_h5(file_path)

        # Overwrite attributes if desired.
        if include_state:
            if len(from_save.keys()) <= 1:
                raise ValueError(
                    "State was not stored in file '{}'"
                    "and cannot be imported".format(file_path)
                )

            is_cupy = ["phase", "amp", "target", "weights", "phase_ff"]
            is_shape = ["shape", "slm_shape"]
            for key in from_save.keys():
                if key != "stats":
                    if key in is_cupy:
                        setattr(self, key, cp.array(from_save[key], dtype=self.dtype, copy=(False if np.__version__[0] == '1' else None)))
                    elif key in is_shape:
                        setattr(self, key, tuple(int(i) for i in from_save[key]))
                    else:
                        setattr(self, key, from_save[key])

        # Overwrite stats, restoring the lists which the update path appends to.
        stats = from_save["stats"]

        stats["method"] = np.asarray(stats["method"]).tolist()
        for key, value in stats["flags"].items():
            stats["flags"][key] = np.asarray(value).tolist()
        for group in stats["stats"].values():
            # Only the outer axis is a list; raw stats hold an array per iteration.
            for key, value in group.items():
                group[key] = list(value)
        if "raw_farfield" in stats:
            stats["raw_farfield"] = list(stats["raw_farfield"])

        self.stats = stats

    # Visualization helper functions.
    @staticmethod
    def _compute_limits(source, epsilon=0, limit_padding=0.1):
        """
        Returns the rectangular region which crops around non-zero pixels in the
        ``source`` image. See :meth:`plot_farfield()`.
        """
        limits = []
        binary = (source > epsilon) & np.logical_not(np.isnan(source))

        # Generated limits on each axis.
        for a in [0, 1]:
            if np.sum(binary) == 0:
                limits.append((0, source.shape[1 - a] - 1))
            else:
                # Collapse the other axis and find the range.
                collapsed = np.where(np.any(binary, axis=a))
                limit = np.array([np.amin(collapsed), np.amax(collapsed)])

                # Add padding.
                padding = int(np.diff(limit)[0] * limit_padding) + 1
                limit += np.array([-padding, padding + 1])

                # Check limits and store.
                limit = np.clip(limit, 0, source.shape[1 - a] - 1)
                limits.append(tuple(limit))

        return limits

    def plot_nearfield(
        self,
        source=None,
        title="",
        padded=False,
        figsize=(8,4),
        cbar=False
    ):
        """
        Plots the amplitude (left) and phase (right) of the nearfield (plane of the SLM).
        The amplitude is assumed (whether uniform, assumed, or measured) while the
        phase is the result of optimization.

        Parameters
        ----------
        source : array_like OR None
            Source to plot. If ``None``, defaults to the current amplitude and phase.
        title : str
            Title of the plots.
        padded : bool
            If ``True``, shows the full computational nearfield of shape :attr:`shape`.
            Otherwise, shows the region at the center of the computational space of
            size :attr:`slm_shape` corresponding to the unpadded SLM.
        figsize : tuple
            Size of the plot.
        cbar : bool
            Whether to add colorbars to the plots. Defaults to ``False``.
        """
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        if source is None:
            amp = self.get_amp()
            phase = self.get_phase()
        else:
            try:
                amp = cp.abs(source).get()
                phase = cp.angle(source).get()
            except Exception:
                amp = np.abs(source)
                phase = np.angle(source)

        if isinstance(amp, float):
            im_amp = axs[0].imshow(
                toolbox.pad(
                    amp * np.ones(self.slm_shape),
                    self.shape if padded else self.slm_shape,
                ),
                vmin=0,
                vmax=amp,
            )
        else:
            im_amp = axs[0].imshow(
                toolbox.pad(amp, self.shape if padded else self.slm_shape),
                vmin=0,
                vmax=np.amax(amp),
            )

        im_phase = axs[1].imshow(
            toolbox.pad(np.mod(phase, 2 * np.pi) / np.pi, self.shape if padded else self.slm_shape),
            vmin=0,
            vmax=2,
            interpolation="none",
            cmap="twilight",
        )

        if len(title) > 0:
            title += ": "

        axs[0].set_title(title + "Amplitude")
        axs[1].set_title(title + "Phase")

        for i, ax in enumerate(axs):
            ax.set_xlabel("SLM $x$ [pix]")
            if i == 0:
                ax.set_ylabel("SLM $y$ [pix]")

        # Add colorbars if desired
        if cbar:
            cax = make_axes_locatable(axs[0]).append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im_amp, cax=cax, orientation="vertical")
            cax = make_axes_locatable(axs[1]).append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im_phase, cax=cax, orientation="vertical", format=r"%1.1f$\pi$")

        fig.tight_layout()
        _slmsuite_plt_show(name="plot_nearfield")

    def plot_farfield(
        self,
        source=None,
        title="",
        limits=None,
        units="knm",
        limit_padding=0.1,
        figsize=(8,4),
        cbar=False,
        axs=None
    ):
        """
        Plots an overview (left) and zoom (right) view of ``source``.

        Parameters
        ----------
        source : array_like OR None
            Should have shape equal to :attr:`shape`.
            If ``None``, defaults to :attr:`amp_ff`.
        title : str
            Title of the plots. If ``"phase"`` is a substring of title, then the source
            is treated as a phase.
        limits : ((float, float), (float, float)) OR None
            :math:`x` and :math:`y` limits for the zoom plot in ``"knm"`` space.
            If None, ``limits`` are autocomputed as the smallest bounds
            that show all non-zero values (plus ``limit_padding``).
            Note that autocomputing on :attr:`target` will perform well,
            as zero values are set to actually be zero. However, doing so on
            computational or experimental outputs (e.g. :attr:`amp_ff`) will likely perform
            poorly, as values in the field deviate slightly from zero and
            artificially expand the ``limits``.
        units : str
            Far-field units for plots (see
            :func:`~slmsuite.holography.toolbox.convert_vector` for options).
            If units requiring a SLM are desired, the attribute :attr:`cameraslm` must be
            filled.
        limit_padding : float
            Fraction of the width and height to expand the limits of the zoom plot by,
            only if the passed ``limits`` is ``None`` (autocompute).
        figsize : tuple
            Size of the plot.
        cbar : bool
            Whether to add colorbars to the plots. Defaults to ``False``.
        axs : matplotlib.axes.Axes OR None
            If provided, uses these axes instead of creating new ones.

        Returns
        -------
        ((float, float), (float, float))
            Used ``limits``, which may be autocomputed. Autocomputed limits are returned
            as integers.
        """
        if source is None:
            source = "amp_ff"

        if isinstance(source, str):
            titles = {
                "amp_ff" : "Farfield Amplitude",
                "phase_ff" : "Farfield Phase",
                "target" : "Target Amplitude",
            }
            if source in titles.keys():
                source_str = source
                source = getattr(self, source)

                if source is None or len(source.shape) == 1:
                    if source_str == "amp_ff":
                        source = self.get_farfield(get=False)
                    else:
                        raise ValueError(f"Could not retrieve source={source_str}")

                if len(title) == 0:
                    title = titles[source_str]

                if limits is None:
                    if len(self.target.shape) == 2:
                        if np == cp:
                            limits = self._compute_limits(self.target, limit_padding=limit_padding)
                        else:
                            limits = self._compute_limits(self.target.get(), limit_padding=limit_padding)
            else:
                raise ValueError(f"Did not recognize source {source}. Must be one of {list(titles.keys())}")

        # Interpret source and convert to numpy for plotting.
        isphase = "phase" in title.lower()
        if isphase:
            try:
                npsource = source.get()
            except Exception:
                npsource = source

            npsource = np.mod(npsource, 2 * np.pi)
        else:
            try:
                npsource = cp.abs(source).get()
            except Exception:
                npsource = np.abs(source)

        # Check units
        if not units in toolbox.BLAZE_UNITS:
            raise ValueError(f"'{units}' is not recognized as a valid blaze unit.")
        if units in toolbox.CAMERA_UNITS:
            raise ValueError(
                f"'{units}' is not a valid unit for plot_farfield() "
                "because of the potential associated rotation."
            )

        # Determine the bounds of the zoom region, padded by limit_padding
        if limits is None:
            limits = self._compute_limits(npsource, limit_padding=limit_padding)
        # Check the limits in case the user provided them.
        limits = [np.clip(np.array(limits[a], dtype=int), 0, npsource.shape[1-a]-1) for a in [0, 1]]
        if any(np.diff(limit)[0] == 0 for limit in limits):
            raise ValueError("Clipped limit has zero length.")

        # Start making the plot
        if axs is None:
            fig, axs = plt.subplots(1, 2, figsize=figsize)
            _show = True
        else:
            _show = False

        # Plot the full target, blurred so single pixels are visible in low res
        b = 2 * int(np.amax(self.shape) / 400) + 1  # FUTURE: fix arbitrary
        npsource_blur = cv2.GaussianBlur(npsource, (b, b), 0)

        full = axs[0].imshow(
            npsource_blur,
            vmin=0, vmax=np.nanmax(npsource),
            cmap=("twilight" if isphase else None),
            interpolation=("none" if isphase else "gaussian")
        )
        if len(title) > 0:
            title += ": "
        axs[0].set_title(title + "Full")

        # Zoom in on our spots in a second plot
        b = 2 * int(np.diff(limits[0])[0] / 200) + 1  # FUTURE: fix arbitrary
        zoom_data = npsource[
            np.ix_(np.arange(limits[1][0], limits[1][1]), np.arange(limits[0][0], limits[0][1]))
        ]
        zoom = axs[1].imshow(
            zoom_data,
            vmin=0, vmax=np.nanmax(zoom_data),
            extent=[limits[0][0], limits[0][1],
                    limits[1][1],limits[1][0]],
            interpolation="none" if b < 2 or isphase else "gaussian",
            cmap=("twilight" if isphase else None)
        )
        axs[1].set_title(title + "Zoom", color="r")
        # Red border (to match red zoom box applied below in "full" img)
        for spine in ["top", "bottom", "right", "left"]:
            axs[1].spines[spine].set_color("r")
            axs[1].spines[spine].set_linewidth(1.5)

        # Helper function: calculate extent for the given units
        try:
            slm = self.cameraslm
        except Exception:
            slm = None
            units = "knm"

        def rebase(ax, img, to_units):
            if to_units != "knm":
                ext_nm = img.get_extent()
                ext_min = np.squeeze(
                    toolbox.convert_vector(
                        [ext_nm[0], ext_nm[-1]],
                        from_units="knm",
                        to_units=to_units,
                        hardware=slm,
                        shape=npsource.shape,
                    )
                )
                ext_max = np.squeeze(
                    toolbox.convert_vector(
                        [ext_nm[1], ext_nm[2]],
                        from_units="knm",
                        to_units=to_units,
                        hardware=slm,
                        shape=npsource.shape,
                    )
                )
                img.set_extent([ext_min[0], ext_max[0], ext_max[1], ext_min[1]])

        # Scale and label plots depending on units
        rebase(axs[0], full, units)
        rebase(axs[1], zoom, units)

        for i, ax in enumerate(axs):
            ax.set_xlabel(toolbox.BLAZE_LABELS[units][0])
            if i == 0:
                ax.set_ylabel(toolbox.BLAZE_LABELS[units][1])

        # Scale aspect; knm might be displaying a non-square array.
        if units == "knm":
            aspect = float(npsource.shape[1]) / float(npsource.shape[0])
        else:
            aspect = 1

        for ax in axs:
            ax.set_facecolor("#FFEEEE")
            ax.set_aspect(aspect)

        # If _cam_points is defined (i.e. is a FeedbackHologram or subclass),
        # plot a yellow rectangle for the extents of the camera
        if hasattr(self, "_cam_points") and self._cam_points is not None:
            _cam_points = self._cam_points.copy()
            _cam_points[0] *= float(npsource.shape[1]) / self.shape[1]
            _cam_points[1] *= float(npsource.shape[0]) / self.shape[0]

            # Check to see if the camera extends outside of knm space.
            plot_slm_fov = (
                np.any(_cam_points[0, :4] < 0)
                or np.any(_cam_points[1, :4] < 0)
                or np.any(_cam_points[0, :4] >= npsource.shape[1])
                or np.any(_cam_points[1, :4] >= npsource.shape[0])
            )

            # If so, plot a labeled green rectangle to show the extents of knm space.
            if plot_slm_fov:
                extent = full.get_extent()
                pix_width = (np.diff(extent[0:2])[0]) / npsource.shape[1]
                rect = plt.Rectangle(
                    np.array(extent[::2]) - pix_width / 2,
                    np.diff(extent[0:2])[0],
                    np.diff(extent[2:])[0],
                    ec="g",
                    fc="none",
                )
                axs[0].add_patch(rect)
                axs[0].annotate(
                    "SLM FoV",
                    (np.mean(extent[:2]), np.max(extent[2:])),
                    c="g",
                    size="small",
                    ha="center",
                    va="top",
                )

            # Convert _cam_points to knm.
            if units != "knm":
                _cam_points = toolbox.convert_vector(
                    _cam_points,
                    from_units="knm",
                    to_units=units,
                    hardware=slm,
                    shape=npsource.shape
                )

            # Plot the labeled yellow rectangle representing the camera.
            axs[0].plot(
                _cam_points[0],
                _cam_points[1],
                c="y",
            )
            axs[0].annotate(
                "Camera FoV",
                (np.mean(_cam_points[0, :4]), np.max(_cam_points[1, :4])),
                c="y",
                size="small",
                ha="center",
                va="top",
            )

            # Determine sensible limits of the field of view.
            if plot_slm_fov:
                dx = np.max(_cam_points[0]) - np.min(_cam_points[0])
                dy = np.max(_cam_points[1]) - np.min(_cam_points[1])
            else:
                dx = dy = 0

            ext = full.get_extent()
            axs[0].set_xlim(
                [
                    min(ext[0], np.min(_cam_points[0]) - dx / 10),
                    max(ext[1], np.max(_cam_points[0]) + dx / 10),
                ]
            )
            axs[0].set_ylim(
                [
                    max(ext[2], np.max(_cam_points[1]) + dy / 10),
                    min(ext[3], np.min(_cam_points[1]) - dy / 10),
                ]
            )

        # Bonus: Plot a red rectangle to show the extents of the zoom region
        if np.diff(limits[0])[0] > 0 and np.diff(limits[1])[0] > 0:
            extent = zoom.get_extent()
            pix_width = (np.diff(extent[0:2])[0]) / np.diff(limits[0])[0]
            rect = plt.Rectangle(
                tuple((np.array(extent[::2]) - pix_width / 2).astype(float)),
                float(np.diff(extent[0:2])[0]),
                float(np.diff(extent[2:])[0]),
                ec="r",
                fc="none",
            )
            axs[0].add_patch(rect)
            axs[0].annotate(
                "Zoom",
                (np.mean(extent[:2]), np.min(extent[2:])),
                c="r",
                size="small",
                ha="center",
                va="bottom",
            )

        # Add colorbar if desired
        if cbar:
            cax = make_axes_locatable(axs[1]).append_axes("right", size="5%", pad=0.05)
            axs[1].get_figure().colorbar(zoom, cax=cax, orientation="vertical")

        if _show:
            plt.tight_layout()
            _slmsuite_plt_show(name="plot_farfield")

        return limits

    def plot_stats(self, stats_dict=None, stat_groups=None, ylim=None, show=False):
        """
        Plots the statistics contained in the given dictionary.

        Parameters
        ----------
        stats_dict : dict OR None
            Stats to plot in dictionary form. If ``None``, defaults to :attr:`stats`.
        stat_groups : list of str OR None
            Which statistics groups to plot. If empty or ``None`` is provided,
            defaults to all groups present in :attr:`stats`.
        ylim : (int, int) OR None
            Allows the user to pass in desired y limits.
            If ``None``, the default y limits are used.
        show : bool
            Whether or not to immediately show the plot. Defaults to ``False``.
        """
        if stats_dict is None:
            stats_dict = self.stats

        _, ax = plt.subplots(1, 1, figsize=(6, 4))

        stats = ["efficiency", "uniformity", "pkpk_err", "std_err"]
        markers = ["o", "o", "s", "D"]
        legendstats = ["inefficiency", "nonuniformity", "pkpk_err", "std_err"]

        niter = np.arange(0, len(stats_dict["method"]))

        if stat_groups is None or len(stat_groups) == 0:
            stat_keys = list(stats_dict["stats"].keys())
        else:
            stat_keys = [str(group) for group in stat_groups]
        dummylines_modes = []

        for ls_num, stat_key in enumerate(stat_keys):
            stat_group = stats_dict["stats"][stat_key]

            for i in range(len(stats)):
                # Invert the stats if it is efficiency or uniformity.
                y = stat_group[stats[i]]
                if i < 2:
                    y = 1 - np.array(y)

                color = "C%d" % ls_num
                line = ax.scatter(
                    niter, y, marker=markers[i], ec=color, fc="None" if i >= 1 else color
                )
                ax.plot(niter, y, c=color, lw=0.5)

                if i == 0:  # Remember the solid lines for the legend.
                    line = ax.plot([], [], c=color)[0]
                    dummylines_modes.append(line)

        # Make the linestyle legend.
        # Inspired from https://stackoverflow.com/a/46214879
        dummylines_keys = []
        for i in range(len(stats)):
            dummylines_keys.append(
                ax.scatter([], [], marker=markers[i], ec="k", fc="None" if i >= 1 else "k")
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Relative Metrics")
        ax.set_title(self.__class__.__name__ + " Statistics")
        ax.set_yscale("log")
        plt.grid()
        try:  # This fails under all nan or other conditions. Fail elegantly.
            plt.tight_layout()
        except Exception:
            pass
        if ylim is not None:
            ax.set_ylim(ylim)

        # Shade fixed_phase. FUTURE: A more general method could be written
        if "fixed_phase" in stats_dict["flags"] and any(stats_dict["flags"]["fixed_phase"]):
            fp = np.concatenate(
                (stats_dict["flags"]["fixed_phase"], [stats_dict["flags"]["fixed_phase"][-1]])
            ) | np.concatenate(
                ([stats_dict["flags"]["fixed_phase"][0]], stats_dict["flags"]["fixed_phase"])
            )
            niter_fp = np.arange(0, len(stats_dict["method"]) + 1)

            ylim = ax.get_ylim()
            poly = ax.fill_between(
                niter_fp - 0.5, ylim[0], ylim[1], where=fp, alpha=0.1, color="b", zorder=-np.inf
            )
            ax.set_ylim(ylim)

            dummylines_keys.append(poly)
            legendstats.append("fixed_phase")

        # Make the color/linestyle legend.
        plt.legend(dummylines_modes + dummylines_keys, stat_keys + legendstats, loc="lower left")

        plt.plot([-.75, len(stats_dict["method"]) - .25], [1,1], alpha=0)

        ax.set_xlim([-0.75, len(stats_dict["method"]) - 0.25])

        if show:
            _slmsuite_plt_show(name="plot_stats")

        return ax
