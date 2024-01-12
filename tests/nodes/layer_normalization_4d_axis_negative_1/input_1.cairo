use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 7892951, sign: true });
    data.append(FP8x23 { mag: 7153170, sign: false });
    data.append(FP8x23 { mag: 6305733, sign: false });
    data.append(FP8x23 { mag: 6298263, sign: true });
    data.append(FP8x23 { mag: 924383, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
