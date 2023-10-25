use array::SpanTrait;

use orion::numbers::NumberTrait;
use orion::operators::tensor::{core::{Tensor, TensorTrait}, math::arithmetic::mul_by_scalar};

/// Cf: NNTrait::gemm docstring
fn gemm<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TAddTensor: Add<Tensor<T>>,
    impl TNumberTrait: NumberTrait<T, MAG>,
    impl TPartialEq: PartialEq<T>,
    impl TMul: Mul<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>
>(
    A: Tensor<T>,
    B: Tensor<T>,
    C: Option<Tensor<T>>,
    alpha: Option<T>,
    beta: Option<T>,
    transA: bool,
    transB: bool
) -> Tensor<T> {
    let alpha: T = if alpha.is_some() {
        alpha.unwrap()
    } else {
        NumberTrait::one()
    };

    let beta: T = if beta.is_some() {
        beta.unwrap()
    } else {
        NumberTrait::one()
    };

    if transA == true {
        let A = A.transpose(array![1, 0].span());
    }

    if transB == true {
        let B = B.transpose(array![1, 0].span());
    }

    match C {
        Option::Some(c) => {
            return mul_by_scalar(@A.matmul(@B), alpha) + mul_by_scalar(@c, beta);
        },
        Option::None(_) => {
            return mul_by_scalar(@A.matmul(@B), alpha);
        }
    }
}
