# tensor.dequantize_linear

```rust
fn dequantize_linear(self: @Tensor<Q>, x_scale: @Tensor<T>, x_zero_point: @Tensor<T>) -> Tensor::<T>;
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

## Type Constraints

u32 tensor, not supported.
fp8x23wide tensor, not supported.
fp16x16wide tensor, not supported.

## Examples

```rust
 use core::array::{ArrayTrait, SpanTrait};
 
 use orion::operators::tensor::{TensorTrait, Tensor, I8Tensor, I32Tensor};
 
 fn dequantize_linear_example() -> Tensor<i32> {
     // We instantiate a 1D Tensor here.
     let x = TensorTrait::<i8>::new(
         shape: array![4].span(),
         data: array![0, 3, 125, 127].span(),
     );
 
     // We instantiate the x_scale here.
     let x_scale = TensorTrait::<i32>::new(
         shape: array![1].span(), data: array![2].span(),
     );
 
     // We instantiate the x_zero_point here.
     let x_zero_point = TensorTrait::<i32>::new(
         shape: array![1].span(), data: array![0].span(),
     );
 
     return x.dequantize_linear(@x_scale, @x_zero_point);
 }
>>> [0, 6, 250, 254]
```
