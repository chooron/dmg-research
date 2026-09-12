# AutoFuse Phase 0 Production Execution & Schedule Synthesis

## 1. Production Execution Profile (Measured Evidence)
- **Batch Size ($B$)**: 2 basins
- **Time Window**: 730 days (365d warmup + 365d scored)
- **Median Optimization Step Time**: 5.64 seconds
- **Production Throughput**: ~638 steps/hour (~1277 basin-windows/hour)
- **Peak VRAM Allocated**: ~61.3 MB (>96% headroom on 12GB VRAM)

## 2. Compilation & Caching Policy
- **TorchInductor Speedup**: 19.66x over eager vmap
- **Numerical Parity**: Verified bitwise/within $10^{-4}$ tolerance
- **Cache Recommendation**: Precompile / cache Inductor kernels to persistent disk (`TORCHINDUCTOR_CACHE_DIR`).

## 3. Candidate Validation Schedules (Arithmetic Projections)
| Nominal Budget | Optimization Steps | Eval Interval | Total Evals | Projected Wall Time | Validation Overhead |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 6h | 3829 | 250 | 15 | 7.2h | 17.0% |
| 6h | 3829 | 500 | 7 | 6.6h | 8.7% |
| 6h | 3829 | 1000 | 3 | 6.2h | 3.9% |
| 12h | 7659 | 250 | 30 | 14.5h | 17.0% |
| 12h | 7659 | 500 | 15 | 13.2h | 9.3% |
| 12h | 7659 | 1000 | 7 | 12.6h | 4.6% |

## 4. Diagnostics & Gradient Stability
- **Gradient Clipping Rate**: 100.0%
- **Parameter Invariant Check**: Inactive parameter heads maintain zero gradient and zero weight drift.

---
*Report generated automatically by Phase 0 Orchestrator at 2026-09-08T14:35:44.200651+00:00*
