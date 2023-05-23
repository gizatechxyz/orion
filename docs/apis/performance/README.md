# Performance

This trait contains a set of functions to increase the performance of your model.

```rust
use orion::performance::performance_i32::performance
// OR
use orion::performance::performance_u32::performance
```

### Data types

Orion supports currently two `performance` types.

| Data type                 | dtype             |
| ------------------------- | ----------------- |
| 32-bit integer (signed)   | `performance_i32` |
| 32-bit integer (unsigned) | `performance_u32` |

| function                                                         | description                                      |
| ---------------------------------------------------------------- | ------------------------------------------------ |
| [`performance.quantize_linear`](performance.quantize\_linear.md) | Quantizes a Tensor using symmetric quantization. |
