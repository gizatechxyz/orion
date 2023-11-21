# tensor.qlinear_concat

```rust 
  qlinear_concat(tensors: Span<Tensor<i8>>, scales: Span<Tensor<T>>, zero_points: Span<Tensor<T>>, y_scale: @Tensor<T>, y_zero_point: @Tensor<T>, axis: usize) -> Tensor::<i8>;
```

Concatenate a list of tensors after dequantizing them with their respective scales and zero_points and returns the quantized result.

## Args

* `tensors`(` Span<Tensor<i8>>,`) - Array of the quantized input tensors.
* `scales`(` Span<Tensor<T>>,`) - Array of the scales of the quantized input tensors.
* `zero_points`(` Span<Tensor<T>>,`) - Arrayof the zero_points of the quantized input tensors.
* `y_scale`(`@Tensor<T>`) - Scale for output.
* `y_zero_point`(`@Tensor<T>`) - Zero point for output.   
* `axis`(`usize`) -  Axis to concat on.

## Panics

* Panic if tensor length is not greater than 1.
* Panics if dimension is not greater than axis.

## Type Constraints

u32 tensor, not supported.
fp8x23wide tensor, not supported.
fp16x16wide tensor, not supported.

## Returns 

A new `Tensor<i8>` concatenated quantized tensor of the dequantized input tensors.

## Example

```rust
use array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, I8Tensor, FP16x16Tensor};
use orion::numbers::{i8, FP16x16, FP16x16Impl, IntegerTrait, FixedTrait};

fn qlinear_concat_example() -> Tensor<i8> {
    let tensor1 = TensorTrait::<
        i8
    >::new(
        shape: array![2, 2].span(),
        data: array![
            IntegerTrait::<i8>::new(5_u8, false),
            IntegerTrait::<i8>::new(5_u8, false),
            IntegerTrait::<i8>::new(5_u8, false),
            IntegerTrait::<i8>::new(5_u8, false),
        ]
            .span(),
    );
    let tensor2 = TensorTrait::<
        i8
    >::new(
        shape: array![2, 2].span(),
        data: array![
            IntegerTrait::<i8>::new(1_u8, false),
            IntegerTrait::<i8>::new(1_u8, false),
            IntegerTrait::<i8>::new(1_u8, false),
            IntegerTrait::<i8>::new(1_u8, false),
        ]
            .span(),
    );

    let tensors = array![tensor1, tensor2].span();

    let tensor1_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(131072, false)].span(),);
    let tensor2_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(262144, false)].span(),);

    let scales = array![tensor1_scale, tensor2_scale].span();

    let tensor1_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(327680, false)].span(),); 
    let tensor2_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),);

    let zero_points = array![tensor1_zero_point, tensor2_zero_point].span();

    let y_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(262144, false)].span(),);

    let y_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(65536, false)].span(),);

    return TensorTrait::qlinear_concat(tensors, scales, zero_points, @y_scale, @y_zero_point, 0);
}

>>> [[1, 1, 1, 1], [2, 2, 2, 2]]  
```
