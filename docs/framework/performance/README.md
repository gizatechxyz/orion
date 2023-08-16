# Performance

This trait contains a set of functions to increase the performance of your model. Performance trait takes two generic parameters:

* `T`, the type of the unquantized tensor.
* `Q`, the type of the quantized tensor.

```rust
use orion::performance::core::PerfomanceTrait;
```

### Data types

Orion supports currently these `performance` types.

<table data-header-hidden><thead><tr><th width="224"></th><th width="170.33333333333331"></th><th></th></tr></thead><tbody><tr><td><code>T</code> Data type</td><td><code>Q</code> Data type</td><td>dtype</td></tr><tr><td>32-bit integer (signed),</td><td>8-bit integer (signed)</td><td><code>PerformanceTrait&#x3C;i32, i8></code></td></tr><tr><td>Fixed point (signed)</td><td>8-bit integer (signed)</td><td><code>PerformanceTrait&#x3C;FixedType, i8></code></td></tr></tbody></table>

| function | description |
| --- | --- |
| [`performance.quantize_linear`](performance.quantize\_linear.md) | Quantizes a Tensor using linear quantization. |
| [`performance.dequantize_linear`](performance.dequantize\_linear.md) | Dequantizes a Tensor using linear dequantization. |

