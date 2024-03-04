use core::array::ArrayTrait;
use core::array::SpanTrait;
use core::option::OptionTrait;

use orion::numbers::{NumberTrait};
use orion::operators::tensor::quantization::dequantize_linear::dequantize_linear;
use orion::operators::tensor::quantization::quantize_linear::quantize_linear;
use orion::operators::tensor::{TensorTrait, Tensor};

fn qlinear_leakyrelu<
    T,
    MAG,
    Q,
    impl TTensor: TensorTrait<T>,
    impl QTensor: TensorTrait<Q>,
    impl QIntoT: Into<Q, T>,
    impl QTensorIntoTTensor: Into<Tensor<Q>, Tensor<T>>,
    impl TAdd: Add<T>,
    impl TSub: Sub<T>,
    impl TDiv: Div<T>,
    impl TMul: Mul<T>,
    impl TTensorAdd: Add<Tensor<T>>,
    impl TTensorSub: Sub<Tensor<T>>,
    impl TTensorMul: Mul<Tensor<T>>,
    impl TTensorDiv: Div<Tensor<T>>,
    impl TPartialOrd: PartialOrd<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TTryInto: TryInto<T, Q>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl QCopy: Copy<Q>,
    impl QDrop: Drop<Q>
>(
    a: @Tensor<Q>, a_scale: @Tensor<T>, a_zero_point: @Tensor<T>, alpha: T, min: T, max: T
) -> Tensor<Q> {
    let mut dequantized_a = dequantize_linear(@(*a), a_scale, a_zero_point);

    let mut result_data = ArrayTrait::<T>::new();
    loop {
        match dequantized_a.data.pop_front() {
            Option::Some(elem) => {
                if *elem < NumberTrait::zero() {
                    result_data.append(*elem * alpha);
                } else {
                    result_data.append(*elem);
                }
            },
            Option::None => { break; }
        };
    };

    return quantize_linear(
        @TensorTrait::new(dequantized_a.shape, result_data.span()), a_scale, a_zero_point, min, max
    );
}
