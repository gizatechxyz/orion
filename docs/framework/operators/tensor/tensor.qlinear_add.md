# tensor.qlinear_add

```rust
    fn qlinear_add(self: @Tensor<i8>, a_scale: @Tensor<T>, a_zero_point: @Tensor<T>, b: @Tensor<i8>, b_scale: @Tensor<T>, b_zero_point: @Tensor<T>, y_scale: @Tensor<T>, y_zero_point: @Tensor<T>) -> Tensor::<i8>;
```

Performs the sum of quantized Tensors

It consumes two quantized input tensors, their scales and zero points, scale and zero point of output, and computes the quantized output. 
The quantization formula is y = saturate((x / y_scale) + y_zero_point).
It perfoms the addition of the two vectors once dequantized, then return the quantization of the result of the addition.
The broadcasting is supported
Scale and zero point must have same shape and the same type. They must be either scalar (per tensor) or N-D tensor (per row for 'a' and per column for 'b'). 
Scalar refers to per tensor quantization whereas N-D refers to per row or per column quantization.

## Args

* `self`(`@Tensor<i8>`) - The first tensor to be additionned (a).
* `a_scale`(`@Tensor<T>`) - Scale for input `a`.
* `a_zero_point`(`@Tensor<T>`) - Zero point for input `a`.
* `b`(`@Tensor<i8>`) - The second tensor to be additionned
* `b_scale`(`@Tensor<T>`) - Scale for input `b`.
* `b_zero_point`(`@Tensor<T>`) - Zero point for input `b`.    
* `y_scale`(`@Tensor<T>`) - Scale for outut.
* `y_zero_point`(`@Tensor<T>`) - Zero point for output.   

## Returns

A new `Tensor<i8>`, containing the quantized result of the addition of the dequantized inputs.

## Type Constraints

u32 tensor, not supported.
fp8x23wide tensor, not supported.
fp16x16wide tensor, not supported.
 
## Example

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, I8Tensor, FP16x16Tensor};
use orion::numbers::{FP16x16, FP16x16Impl, FixedTrait};


fn qlinear_add_example() -> Tensor<i8> {    
    let a = TensorTrait::<
        i8
    >::new(
        shape: array![2, 3].span(),
        data: array![6, 6, 6, 11, 11, 11].span(),
    );

    // As the operator supports broadcasting shapes [1, 3] and [2, 3] are compatible
    let b = TensorTrait::<i8>::new(
        shape: array![1, 3].span(),
        data: array![40, 40, 40].span(),
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
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(655360, false)].span(),);
    let y_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(65536, true)].span(),);

    return a
        .qlinear_add(
            @a_scale, @a_zero_point, @b, @b_scale, @b_zero_point, @y_scale, @y_zero_point
        );
}        

>>> [[1, 1, 1], [2, 2, 2]]
```
