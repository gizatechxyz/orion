# Performance

This module contains a set of functions to increase the performance of your model.

```rust
use onnx_cairo::performance::performance_i32::performance
// OR
use onnx_cairo::performance::performance_u32::performance
```

### Data types

ONNX-Cairo supports currently two `performance` types.

| Data type                 | dtype             |
| ------------------------- | ----------------- |
| 32-bit integer (signed)   | `performance_i32` |
| 32-bit integer (unsigned) | `performance_u32` |

| function                                    | description                                      |
| ------------------------------------------- | ------------------------------------------------ |
| [`quantize_linear`](linear-quantization.md) | Quantizes a Tensor using symmetric quantization. |
