# Performance

This trait contains a set of functions to increase the performance of your model. 
Performance trait takes two generic parameters: 
- `T`, the type of the unquantized tensor.
- `Q`, the type of the quantized tensor.

```rust
use orion::performance::core::PerfomanceTrait;
```

### Data types

Orion supports currently these `performance` types.

| `T` Data type            | `Q` Data type          | dtype                             |
| ------------------------ | ---------------------- | --------------------------------- |
| 32-bit integer (signed), | 8-bit integer (signed) | `PerformanceTrait<i32, i8>`       |
| Fixed point  (signed)    | 8-bit integer (signed) | `PerformanceTrait<FixedType, i8>` |


| function | description |
| --- | --- |
| [`performance.quantize_linear`](performance.quantize\_linear.md) | Quantizes a Tensor using linear quantization. |
| [`performance.dequantize_linear`](performance.dequantize\_linear.md) | Dequantizes a Tensor using linear dequantization. |

