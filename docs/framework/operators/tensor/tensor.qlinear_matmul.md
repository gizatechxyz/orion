# tensor.qlinear_matmul

```rust
    fn qlinear_matmul(self: @Tensor<i8>, a_scale: @Tensor<T>, a_zero_point: @Tensor<T>, b: @Tensor<i8>, b_scale: @Tensor<T>, b_zero_point: @Tensor<T>, y_scale: @Tensor<T>, y_zero_point: @Tensor<T>) -> Tensor::<i8>;
```

Multiplies quantized Tensors

It consumes two quantized input tensors, their scales and zero points, scale and zero point of output, and computes the quantized output. 
The quantization formula is y = saturate((x / y_scale) + y_zero_point).
It perfoms the multiplication of the two vectors once dequantized. If either argument is N-D, N > 2, it is treated as a stack of matrices residing in the last two indexes.
Then return the quantization of the result of the multiplication.
Scale and zero point must have same shape and the same type. They must be either scalar (per tensor) or N-D tensor (per row for 'a' and per column for 'b'). 
Scalar refers to per tensor quantization whereas N-D refers to per row or per column quantization.

## Args

* `self`(`@Tensor<i8>`) - The first tensor to be multiplied (a).
* `a_scale`(`@Tensor<T>`) - Scale for input `a`.
* `a_zero_point`(`@Tensor<T>`) - Zero point for input `a`.
* `b`(`@Tensor<i8>`) - The second tensor to be multiplied
* `b_scale`(`@Tensor<T>`) - Scale for input `b`.
* `b_zero_point`(`@Tensor<T>`) - Zero point for input `b`.    
* `y_scale`(`@Tensor<T>`) - Scale for output.
* `y_zero_point`(`@Tensor<T>`) - Zero point for output.   

## Returns

A new `Tensor<i8>`, containing the quantized result of the multiplication of the dequantized inputs.

## Type Constraints

u32 tensor, not supported.
fp8x23wide tensor, not supported.
fp16x16wide tensor, not supported.
 
## Example

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, I8Tensor, FP16x16Tensor};
use orion::numbers::{i8, FP16x16, FP16x16Impl, IntegerTrait, FixedTrait};
fn qlinear_matmul_example() -> Tensor<i8> {    
    let a = TensorTrait::<
        i8
    >::new(
        shape: array![2, 3].span(),
        data: array![
            IntegerTrait::<i8>::new(3_u8, false),
            IntegerTrait::<i8>::new(4_u8, false),
            IntegerTrait::<i8>::new(5_u8, false),
            IntegerTrait::<i8>::new(2_u8, false),
            IntegerTrait::<i8>::new(4_u8, false),
            IntegerTrait::<i8>::new(3_u8, false)
        ]
            .span(),
    );
    let b = TensorTrait::<
        i8
    >::new(
        shape: array![3, 1].span(),
        data: array![
            IntegerTrait::<i8>::new(4_u8, false),
            IntegerTrait::<i8>::new(8_u8, false),
            IntegerTrait::<i8>::new(4_u8, false)
        ]
            .span(),
    );

    let a_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(131072, false)].span(),);
    let a_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(65536, false)].span(),);
    let b_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(16384, false)].span(),);
    let b_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(0, false)].span(),);

    let y_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(393216, false)].span(),);
    let y_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(655360, false)].span(),);

    return a
        .qlinear_matmul(
            @a_scale, @a_zero_point, @b, @b_scale, @b_zero_point, @y_scale, @y_zero_point
        );
}        
>>> [14, 13]
```
