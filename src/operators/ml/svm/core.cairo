use orion::numbers::NumberTrait;
use orion::numbers::{FP16x16, FP16x16Impl, FP32x32, FP32x32Impl, FixedTrait};
use orion::operators::tensor::{
    TensorTrait, Tensor, I8Tensor, I32Tensor, U32Tensor, FP16x16Tensor, BoolTensor
};
use orion::utils::get_row;

#[derive(Copy, Drop)]
enum KERNEL_TYPE {
    LINEAR,
    POLY,
    RBF,
    SIGMOID,
}

fn kernel_dot<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +Add<T>,
    +TensorTrait<T>,
    +AddEq<T>,
    +Mul<T>,
    +Neg<T>,
    +Sub<T>,
>(
    kernel_params: Span<T>, pA: Span<T>, pB: Span<T>, kernel: KERNEL_TYPE
) -> T {
    let s = match kernel {
        KERNEL_TYPE::LINEAR => sv_dot(pA, pB),
        KERNEL_TYPE::POLY => {
            let mut s = sv_dot(pA, pB);
            s = s * *kernel_params.at(0) + *kernel_params.at(1);
            s.pow(*kernel_params.at(2))
        },
        KERNEL_TYPE::RBF => {
            let mut s = squared_diff(pA, pB);
            NumberTrait::exp(-*kernel_params.at(0) * s)
        },
        KERNEL_TYPE::SIGMOID => {
            let mut s = sv_dot(pA, pB);
            s = s * *kernel_params.at(0) + *kernel_params.at(1);
            NumberTrait::tanh(s)
        },
    };

    s
}


fn sv_dot<
    T, MAG, +Drop<T>, +Copy<T>, +NumberTrait<T, MAG>, +Add<T>, +TensorTrait<T>, +AddEq<T>, +Mul<T>,
>(
    pA: Span<T>, pB: Span<T>
) -> T {
    let mut i = 0;
    let mut sum = NumberTrait::zero();
    while i != pA.len() {
        sum = sum + *pA.at(i) * *pB.at(i);
        i += 1;
    };

    sum
}

fn squared_diff<
    T,
    MAG,
    +Drop<T>,
    +Copy<T>,
    +NumberTrait<T, MAG>,
    +Add<T>,
    +TensorTrait<T>,
    +AddEq<T>,
    +Mul<T>,
    +Sub<T>,
>(
    pA: Span<T>, pB: Span<T>
) -> T {
    let mut i = 0;
    let mut sum = NumberTrait::zero();
    while i != pA
        .len() {
            sum = sum + (*pA.at(i) - *pB.at(i)).pow(NumberTrait::one() + NumberTrait::one());
            i += 1;
        };

    sum
}
