use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 4256, sign: true });
    data.append(FP16x16 { mag: 12347, sign: true });
    data.append(FP16x16 { mag: 12223, sign: true });
    data.append(FP16x16 { mag: 35664, sign: false });
    data.append(FP16x16 { mag: 47729, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
