"""
benchmark.py — Compare CPU vs GPU propagation speed
----------------------------------------------------
Run with:  python benchmark.py
"""

import numpy as np
import matplotlib.pyplot as plt
from openbeam.backend import set_backend, to_numpy
from openbeam.propagation_gpu import AngularSpectrumPropagator

GRID_SIZES = [256, 512, 1024, 2048]
WAVELENGTH = 532e-9   # 532 nm
PHYS_SIZE = 10e-3     # 10 mm
Z = 0.5               # 50 cm propagation
N_RUNS = 20


def run_benchmark(backend: str):
    set_backend(backend)
    times = []
    for N in GRID_SIZES:
        prop = AngularSpectrumPropagator(
            grid_size=N,
            physical_size=PHYS_SIZE,
            wavelength=WAVELENGTH,
            z=Z,
        )
        result = prop.benchmark(n_runs=N_RUNS)
        times.append(result["avg_ms"])
        print(f"  [{backend.upper():4s}]  N={N:4d}  →  {result['avg_ms']:7.2f} ms/call")
    return times


print("=" * 50)
print("openbeam  GPU acceleration benchmark")
print("=" * 50)

print("\n▶  CPU (NumPy)")
cpu_times = run_benchmark("cpu")

print("\n▶  GPU (CuPy)")
gpu_times = run_benchmark("cuda")

# ── Plot ──────────────────────────────────────────────────────────────
speedups = [c / g if g > 0 else 0 for c, g in zip(cpu_times, gpu_times)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(GRID_SIZES, cpu_times, "o-", label="CPU (NumPy)", color="#4C72B0", linewidth=2)
ax1.plot(GRID_SIZES, gpu_times, "s-", label="GPU (CuPy)", color="#DD8452", linewidth=2)
ax1.set_xlabel("Grid size (N × N)")
ax1.set_ylabel("Average propagation time (ms)")
ax1.set_title("ASM Propagation: CPU vs GPU")
ax1.legend()
ax1.set_yscale("log")
ax1.grid(True, which="both", alpha=0.3)

ax2.bar(GRID_SIZES, speedups, color="#55A868", width=[s * 0.6 for s in GRID_SIZES])
ax2.set_xlabel("Grid size (N × N)")
ax2.set_ylabel("Speedup (×)")
ax2.set_title("GPU Speedup over CPU")
ax2.grid(True, axis="y", alpha=0.3)
for i, (n, s) in enumerate(zip(GRID_SIZES, speedups)):
    ax2.text(n, s + 0.2, f"{s:.1f}×", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig("benchmark_results.png", dpi=150)
plt.show()
print("\nPlot saved to benchmark_results.png")
