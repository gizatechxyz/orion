# tensor.qlinear_leakyrelu

```rust
    fn qlinear_leakyrelu(self: @Tensor<i8>, a_scale: @Tensor<T>, a_zero_point: @Tensor<T>, alpha: T) -> Tensor::<i8>;
```

Applies the Leaky Relu operator to a quantized Tensor

QLinar LeakyRelu takes as input a quantized Tensor, its scale and zero point and an scalar alpha, and produces one output data (a quantized Tensor)
where the function `f(x) = alpha * x for x < 0, f(x) = x for x >= 0`, is applied to the data tensor elementwise.
The quantization formula is y = saturate((x / y_scale) + y_zero_point).
Scale and zero point must have same shape and the same type. They must be either scalar (per tensor) or N-D tensor (per row for 'a' and per column for 'b'). 
Scalar refers to per tensor quantization whereas N-D refers to per row or per column quantization.

## Args

* `self`(`@Tensor<i8>`) - The first tensor to be multiplied (a).
* `a_scale`(`@Tensor<T>`) - Scale for input `a`.
* `a_zero_point`(`@Tensor<T>`) - Zero point for input `a`.
* `alpha`(`T`) - The factor multiplied to negative elements.

## Returns

A new `Tensor<i8>`, containing result of the Leaky Relu.

## Type Constraints

u32 tensor, not supported.
fp8x23wide tensor, not supported.
fp16x16wide tensor, not supported.
bool tensor, not supported.
 
## Example

```rust

use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, I8Tensor, FP16x16Tensor};
use orion::numbers::{i8, FP16x16, FP16x16Impl, IntegerTrait, FixedTrait};


fn qlinear_leakyrelu_example() -> Tensor<i8> {
    let a = TensorTrait::<
        i8
    >::new(
        shape: array![2, 3].span(),
        data: array![
            IntegerTrait::<i8>::new(10_u8, true),
            IntegerTrait::<i8>::new(10_u8, true),
            IntegerTrait::<i8>::new(10_u8, true),
            IntegerTrait::<i8>::new(10_u8, false),
            IntegerTrait::<i8>::new(10_u8, false),
            IntegerTrait::<i8>::new(10_u8, false)
        ]
            .span(),
    );
    
    let a_scale = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(327680, false)].span(),);
    let a_zero_point = TensorTrait::<
        FP16x16
    >::new(shape: array![1].span(), data: array![FixedTrait::<FP16x16>::new(131072, false)].span(),);

    let alpha = FixedTrait::<FP16x16>::new(655360, false);

    return = a
        .qlinear_leakyrelu(
            @a_scale, @a_zero_point, alpha
        );
}

>>> [[-118, -118, -118], [10, 10, 10]]
