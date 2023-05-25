# performance.quantize_linear_from_fp

```rust
fn quantize_linear_from_fp(self: @Tensor<FixedType>) -> Tensor<T>;
```

Quantizes a FixedType Tensor using symmetric quantization.

This is an 8-bit linear quantization of a tensor. This method allows tensors to be stored at lower bitwidths than those of fixed-point precision.

During quantization, the unquantized values are mapped to an 8 bit quantization space of the form:

`quantized_value = value / scale`

`scale` is a positive number used to map the unquantized numbers to a quantization space. It is calculated as follows in symmetric quantization:

```
scale = max(abs(data_range_max), abs(data_range_min)) * 2 / (quantization_range_max - quantization_range_min)
```

## Args

* `tensor`(`@Tensor<FixedType>`) - The input FixedType tensor.

## Returns

A new `Tensor<T>` with the same shape as the input tensor, containing the quantized values.

## Examples

```rust
use orion::performance::core::PerfomanceTrait;
use orion::performance::implementations::impl_performance_i32;

fn quantize_linear_example() -> Tensor<i32> {
// We instantiate a FixedType 2D Tensor here.
// [[838860800, 1258291200, 1677721600],[-838860800, -1258291200, -1677721600]]
let tensor = fp_tensor_3x2_helper();

// We can call `quantize_linear_from_fp` function as follows.
return PerfomanceTrait::quantize_linear_from_fp(@tensor);
}
>>> [[63, 95, 127],[-63, -95, -127]]
```
