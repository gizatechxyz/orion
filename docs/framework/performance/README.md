# Performance

This trait contains a set of functions to increase the performance of your model.

```rust
use orion::performance::performance_i32::performance
// OR
use orion::performance::performance_u32::performance
```

### Data types

Orion supports currently these `performance` types.

| Data type                 | dtype                              |
| ------------------------- | ---------------------------------- |
| 32-bit integer (signed)   | `PerformanceTrait<i32, i32>`       |
| 32-bit integer (unsigned) | `PerformanceTrait<u32, u32>`       |
| Fixed point  (signed)     | `PerformanceTrait<FixedType, i32>` |
| Fixed point  (signed)     | `PerformanceTrait<FixedType, u32>` |


| function | description |
| --- | --- |
| [`performance.quantize_linear`](performance.quantize\_linear.md) | Quantizes a Tensor using linear quantization. |
| [`performance.dequantize_linear`](performance.dequantize\_linear.md) | Dequantizes a Tensor using linear dequantization. |

