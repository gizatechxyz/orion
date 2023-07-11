# performance.dequantize_linear

```rust
fn dequantize_linear(self: @Tensor<T>, x_scale: @Tensor<T>, x_zero_point: @Tensor<T>) -> Tensor::<T>;
```

Dequantizes a Tensor using linear dequantization.

The linear dequantization operator. It consumes a quantized tensor, a scale, and a zero point to compute
the full precision tensor. The dequantization formula is y = (x - x_zero_point) * x_scale. x_scale and
x_zero_point must have same shape, and can be either a scalar for per-tensor / per layer quantization,
or a 1-D tensor for per-axis quantization.

## Args

* `self`(`@Tensor<T>`) - The input tensor.
* `x_scale`(`@Tensor<T>`) - Scale for input `x`.
* `x_zero_point`(`@Tensor<T>`) - Zero point for input `x`.

## Returns

A new `Tensor<T>` with the same shape as the input tensor, containing the dequantized values.

## Examples

```rust
use orion::performance::core::PerfomanceTrait;
use orion::performance::implementations::impl_performance_i32::Performance_i32_i8;

fn quantize_linear_example() -> Tensor<i32> {
// We instantiate a 1D quantizes Tensor here.
// [0, 3, 128, 255]
let x = i32_tensor_1D_helper();

// We instantiate the x_scale here.
let mut shape = ArrayTrait::<usize>::new();
shape.append(1);
let mut data = ArrayTrait::<i32>::new();
data.append(IntegerTrait::new(2, false));
let extra = Option::<ExtraParams>::None(());
let x_scale = TensorTrait::new(shape.span(), data.span(), extra);

// We instantiate the x_zero_point here.
let mut shape = ArrayTrait::<usize>::new();
shape.append(1);
let mut data = ArrayTrait::<i32>::new();
data.append(IntegerTrait::new(128, false));
let extra = Option::<ExtraParams>::None(());
let x_zero_point = TensorTrait::new(shape.span(), data.span(), extra);

// We can call `dequantize_linear` function as follows.
return x.dequantize_linear(@x_scale, @x_zero_point);
}
>>> [-256, -250, 0, 254]
```
