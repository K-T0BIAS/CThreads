# Math and linalg

`cthreads.linalg` provides array / tensor ops for multithreaded workloads. On SIMD-capable CPUs (AVX2), contiguous kernels use vectorized paths for ewise, compare, mask, dot, cross, and tiled GEMM.

## Performance vs NumPy

Heavy-suite wall-clock comparison (`out.csv`): best-of-11, speedup = NumPy ms ÷ cthreads ms (**>1 means cthreads is faster**).

| | f32 | f64 |
|---|---|---|
| Median speedup (all cases) | 0.82× | 0.81× |
| Cases where cthreads wins | 26 / 88 | 25 / 88 |
| Ewise / compare (typical) | ~0.75–0.85× | ~0.65–0.85× |
| Large GEMM serial vs OpenBLAS | ~0.10× | ~0.10× |
| Large GEMM `parallel=True` vs OpenBLAS | ~0.41–0.56× | ~0.47–0.56× |

![cthreads.linalg vs NumPy](../__ressources/linalg_vs_numpy.svg)

| Family | f32 median | f64 median |
|---|---:|---:|
| Ewise 1D | 0.83× | 0.84× |
| Ewise 2D | 0.81× | 0.80× |
| Inplace | 0.77× | 0.77× |
| Compare | 0.74× | 0.65× |
| Mask | **1.45×** | **1.41×** |
| Dot | **7.29×** | **4.55×** |
| Cross | **3.59×** | **3.80×** |
| Matmul (serial) | 0.12× | 0.13× |
| Matmul (`parallel=True`) | ~0.41–0.56× | ~0.47–0.56× |

**Strong:** row-wise dot, batched cross, mask select/count/fill.  
**Close:** contiguous ewise / compare (AVX paths in the same league as NumPy).  
**Matmul:** default `a.matmul(b)` stays single-threaded so it does not fight the thread pool (~0.10× vs vendor BLAS on large squares). Opt in with `a.matmul(b, parallel=True)` to panel-parallelize large GEMMs (`M*N*K` above ~67e6, using ~80% of hardware threads) — typically ~4–5× faster than serial and about half of NumPy/OpenBLAS on the heavy 2D shapes below.

### Matmul serial vs `parallel=True` (heavy suite)

Same NumPy `a @ b` baseline for both columns. Speedup = NumPy ms ÷ cthreads ms (best-of).

| Shape | f32 serial | f32 parallel | f64 serial | f64 parallel |
|---|---:|---:|---:|---:|
| 1024³ | 0.14× | **0.41×** | 0.14× | **0.47×** |
| 2048×1024×2048 | 0.13× | **0.55×** | 0.12× | **0.56×** |
| 2048³ | 0.10× | **0.49×** | 0.10× | **0.52×** |
| B8:256³ | 0.74× | 0.75× | 0.38× | 0.36× |
| B4:512³ | 0.24× | **0.37×** | 0.16× | 0.20× |

Batched 256³ barely moves: each slab is under / near the `M*N*K` gate and has few `Nc` panels, so `parallel=True` often stays serial.

Layout `flatten` / `reshape` can look 100–400× because cthreads returns a view while this bench materializes a NumPy copy; they are omitted from the chart. Transpose alone is about 1–2×.

<details>
<summary>Highlight cases (largest shape, best-of ms)</summary>

| Op | Dtype | Shape | NumPy | cthreads | Speedup |
|---|---|---|---:|---:|---:|
| ewise1d.add | f32 | 4M | 2.20 | 2.87 | 0.77× |
| compare.gt | f32 | 4M | 0.99 | 1.31 | 0.76× |
| mask.select | f32 | 4M | 16.4 | 12.1 | **1.36×** |
| mask.fill | f32 | 4M | 2.82 | 0.56 | **5.03×** |
| mask.count | f32 | 4M | 0.077 | 0.054 | **1.44×** |
| dot.rows | f32 | 1024×4096 | 4.63 | 0.60 | **7.73×** |
| cross.batched | f32 | 262k×3 | 1.77 | 0.64 | **2.78×** |
| matmul.2d | f32 | 2048³ | 19.6 | 197 | 0.10× |
| matmul.2d_parallel | f32 | 2048³ | 19.5 | 39.8 | **0.49×** |
| matmul.2d_parallel | f64 | 2048³ | 42.0 | 80.3 | **0.52×** |
| layout.transpose | f32 | 2048² | 26.2 | 12.5 | **2.11×** |

</details>

### Reproduce

```bash
python docs/demos/linalg_benchmark/bench_linalg_vs_numpy.py --dtype both --suite heavy --csv out.csv
# matmul only (serial + parallel rows):
python docs/demos/linalg_benchmark/bench_linalg_vs_numpy.py --dtype both --suite heavy --ops matmul --csv out.csv
```

More detail: [linalg benchmark README](../demos/linalg_benchmark/README.md).
