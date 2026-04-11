"""
propagation_gpu.py — GPU-accelerated Angular Spectrum Method (ASM)
-------------------------------------------------------------------
Drop-in replacements / extensions for openbeam's propagation routines.

All heavy arrays (FFT, transfer function, field) live entirely on the
device (GPU if CuPy is active, RAM if NumPy).  Only call ``to_numpy()``
when you need results on the CPU (e.g. for plotting).

Example
-------
    from openbeam.backend import set_backend, to_numpy
    from openbeam.propagation_gpu import AngularSpectrumPropagator

    set_backend("cuda")   # switch to GPU  (or "cpu" for NumPy)

    prop = AngularSpectrumPropagator(
        grid_size=1024,
        physical_size=10e-3,   # 10 mm window
        wavelength=532e-9,     # 532 nm green laser
        z=0.5,                 # propagate 0.5 m
    )

    # Build an input field (e.g. Gaussian beam)
    field_in = prop.gaussian_beam(waist=1e-3)

    # Propagate
    field_out = prop.propagate(field_in)

    # Bring result to CPU for plotting
    import matplotlib.pyplot as plt
    intensity = to_numpy(xp.abs(field_out) ** 2)
    plt.imshow(intensity, cmap="inferno")
    plt.show()
"""

from __future__ import annotations
import time
import numpy as np  # always available for type hints / fallback

from openbeam.backend import get_array_module, to_numpy, current_backend


class AngularSpectrumPropagator:
    """
    Vectorised, backend-agnostic Angular Spectrum Method propagator.

    The transfer function H is pre-computed once at construction so that
    repeated calls to ``propagate()`` cost only two FFTs + one multiply.

    Parameters
    ----------
    grid_size : int
        Number of pixels along each axis (square grid assumed).
    physical_size : float
        Side length of the simulation window in metres.
    wavelength : float
        Optical wavelength in metres.
    z : float
        Propagation distance in metres.
    bandlimit : bool
        If True, apply the Matsushima (2009) band-limiting filter to
        suppress aliased evanescent components.  Recommended for z/size > 1.
    """

    def __init__(
        self,
        grid_size: int = 512,
        physical_size: float = 10e-3,
        wavelength: float = 532e-9,
        z: float = 0.1,
        bandlimit: bool = True,
    ) -> None:
        self.N = grid_size
        self.L = physical_size
        self.lam = wavelength
        self.z = z
        self.bandlimit = bandlimit

        self._H = None          # transfer function (device array)
        self._coords = None     # cached spatial coords
        self._build_transfer_function()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def propagate(self, field):
        """
        Propagate a complex field by distance ``self.z``.

        Parameters
        ----------
        field : array-like, shape (N, N), complex
            Input complex amplitude.  Can be a NumPy or CuPy array.

        Returns
        -------
        array, shape (N, N), complex
            Output complex amplitude on the same device as ``field``.
        """
        xp = get_array_module()
        field = xp.asarray(field)
        F = xp.fft.fftshift(xp.fft.fft2(field))
        F_prop = F * self._H
        return xp.fft.ifft2(xp.fft.ifftshift(F_prop))

    def propagate_z_stack(self, field, z_values):
        """
        Propagate to multiple distances in a single call.

        Useful for 3-D visualisation without re-doing the FFT each time.

        Parameters
        ----------
        field : array-like, shape (N, N), complex
        z_values : sequence of float
            Distances to propagate to (metres).

        Returns
        -------
        stack : array, shape (len(z_values), N, N), complex
        """
        xp = get_array_module()
        field = xp.asarray(field)
        F = xp.fft.fftshift(xp.fft.fft2(field))

        results = []
        for z in z_values:
            H_z = self._compute_transfer_function(z)
            out = xp.fft.ifft2(xp.fft.ifftshift(F * H_z))
            results.append(out)

        return xp.stack(results, axis=0)

    def gaussian_beam(self, waist: float, offset_x: float = 0.0, offset_y: float = 0.0):
        """
        Generate a Gaussian beam input field.

        Parameters
        ----------
        waist : float
            1/e² beam radius in metres.
        offset_x, offset_y : float
            Centre offset in metres (default: centred).

        Returns
        -------
        array, shape (N, N), complex128
        """
        xp = get_array_module()
        x, y = self._get_coords()
        r2 = (x - offset_x) ** 2 + (y - offset_y) ** 2
        field = xp.exp(-r2 / waist ** 2).astype(xp.complex128)
        return field

    def plane_wave(self, angle_x: float = 0.0, angle_y: float = 0.0):
        """
        Tilted plane wave input field.

        Parameters
        ----------
        angle_x, angle_y : float
            Tilt angles in radians.
        """
        xp = get_array_module()
        x, y = self._get_coords()
        k = 2 * np.pi / self.lam
        phase = k * (x * np.sin(angle_x) + y * np.sin(angle_y))
        return xp.exp(1j * phase).astype(xp.complex128)

    def apply_lens(self, field, focal_length: float):
        """
        Apply a thin-lens phase mask to the field.

        Parameters
        ----------
        field : array, shape (N, N), complex
        focal_length : float
            Focal length in metres (positive = converging).
        """
        xp = get_array_module()
        x, y = self._get_coords()
        k = 2 * np.pi / self.lam
        lens_phase = xp.exp(-1j * k / (2 * focal_length) * (x ** 2 + y ** 2))
        return field * lens_phase

    def apply_circular_aperture(self, field, radius: float):
        """
        Apply a hard circular aperture.

        Parameters
        ----------
        radius : float
            Aperture radius in metres.
        """
        xp = get_array_module()
        x, y = self._get_coords()
        mask = (x ** 2 + y ** 2) <= radius ** 2
        return field * mask.astype(field.dtype)

    def benchmark(self, n_runs: int = 10):
        """
        Measure average propagation time over ``n_runs`` calls.

        Returns
        -------
        dict with keys: backend, avg_ms, total_ms, n_runs
        """
        xp = get_array_module()
        field = self.gaussian_beam(waist=self.L / 10)

        # warm-up
        _ = self.propagate(field)
        if current_backend() == "cuda":
            xp.cuda.Stream.null.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_runs):
            result = self.propagate(field)
        if current_backend() == "cuda":
            xp.cuda.Stream.null.synchronize()
        total = (time.perf_counter() - t0) * 1000  # ms

        return {
            "backend": current_backend(),
            "n_runs": n_runs,
            "total_ms": round(total, 2),
            "avg_ms": round(total / n_runs, 2),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_transfer_function(self):
        """Pre-compute H for self.z."""
        self._H = self._compute_transfer_function(self.z)

    def _compute_transfer_function(self, z: float):
        """
        Compute the ASM transfer function H(fx, fy; z).

        H = exp(i * kz * z)  for propagating waves
        H = 0                 for evanescent waves  (|k_perp| > k)
        """
        xp = get_array_module()

        dx = self.L / self.N
        k = 2 * np.pi / self.lam

        # Spatial-frequency axes (cycles/m)
        fx = xp.fft.fftshift(xp.fft.fftfreq(self.N, d=dx))
        fy = xp.fft.fftshift(xp.fft.fftfreq(self.N, d=dx))
        FX, FY = xp.meshgrid(fx, fy)

        # Square of transverse spatial frequency
        f_sq = FX ** 2 + FY ** 2
        k_sq = (1.0 / self.lam) ** 2

        # Evanescent mask (propagating waves only)
        prop_mask = f_sq <= k_sq

        # z-component of k-vector
        kz = xp.where(prop_mask, 2 * np.pi * xp.sqrt(xp.maximum(k_sq - f_sq, 0.0)), 0.0)

        H = xp.where(prop_mask, xp.exp(1j * kz * z), xp.zeros_like(kz, dtype=complex))

        if self.bandlimit:
            H = H * self._bandlimit_filter(FX, FY, z)

        return H.astype(xp.complex128)

    def _bandlimit_filter(self, FX, FY, z: float):
        """
        Matsushima & Shimobaba (2009) band-limiting filter.
        Prevents aliasing artifacts at large propagation distances.
        """
        xp = get_array_module()

        # Critical spatial frequency limits
        fx_limit = 1.0 / xp.sqrt((2 * self.lam * z / self.L) ** 2 + 1) / self.lam
        fy_limit = fx_limit  # square grid

        filter_x = (xp.abs(FX) <= fx_limit).astype(float)
        filter_y = (xp.abs(FY) <= fy_limit).astype(float)
        return filter_x * filter_y

    def _get_coords(self):
        """Return (x, y) meshgrids in metres, cached after first call."""
        if self._coords is None:
            xp = get_array_module()
            coords_1d = xp.linspace(-self.L / 2, self.L / 2, self.N, endpoint=False)
            self._coords = xp.meshgrid(coords_1d, coords_1d)
        return self._coords