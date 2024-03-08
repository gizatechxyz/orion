use core::debug::PrintTrait;

use orion::numbers::NumberTrait;
use orion::numbers::{U32IntoI32, I32IntoU32, I32Div, I32Number};
use orion::operators::tensor::{TensorTrait, Tensor, U32Tensor,};
use orion::operators::vec::{NullableVec, NullableVecImpl};
use orion::operators::tensor::implementations::tensor_u32::{U32TensorSub};
use orion::operators::tensor::implementations::tensor_i32::{I32TensorTryIntoU32Tensor};
use orion::operators::tensor::implementations::tensor_i8::{I8TensorTryIntoU32Tensor};

use orion::operators::nn::functional::conv::{conv, AUTO_PAD};

/// Cf: NNTrait::conv_integer docstring
fn conv_integer<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TAdd: Add<T>,
    impl TSub: Sub<T>,
    impl TDiv: Div<T>,
    impl TMul: Mul<T>,
    impl TTensorMul: Mul<Tensor<T>>,
    impl TTensorSub: Sub<Tensor<T>>,
    impl TTensorDiv: Div<Tensor<T>>,
    impl TPartialOrd: PartialOrd<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    +PrintTrait<T>,
    +TryInto<Tensor<T>, Tensor<usize>>
>(
    X: @Tensor<T>,
    W: @Tensor<T>,
    X_zero_point: Option<@Tensor<T>>,
    W_zero_point: Option<@Tensor<T>>,
    auto_pad: Option<AUTO_PAD>,
    dilations: Option<Span<usize>>,
    group: Option<usize>,
    kernel_shape: Option<Span<usize>>,
    pads: Option<Span<usize>>,
    strides: Option<Span<usize>>,
) -> Tensor<usize> {
    assert((*X).shape.len() >= 3, 'X must have at least 3 dim');
    let X: Tensor<usize> = match X_zero_point {
        Option::Some(X_zero_point) => {
            ((*X).try_into().unwrap() - (*X_zero_point).try_into().unwrap())
        },
        Option::None => { (*X).try_into().unwrap() }
    };

    let W: Tensor<usize> = match W_zero_point {
        Option::Some(W_zero_point) => {
            ((*W).try_into().unwrap() - (*W_zero_point).try_into().unwrap())
        },
        Option::None => { (*W).try_into().unwrap() }
    };

    return conv(@X, @W, Option::None, auto_pad, dilations, group, kernel_shape, pads, strides);
}
