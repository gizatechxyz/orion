use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 93216, sign: true });
    data.append(FP16x16 { mag: 37735, sign: false });
    data.append(FP16x16 { mag: 93039, sign: false });
    data.append(FP16x16 { mag: 65618, sign: false });
    data.append(FP16x16 { mag: 52063, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
