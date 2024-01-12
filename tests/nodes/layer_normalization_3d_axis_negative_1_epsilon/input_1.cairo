use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 9463, sign: false });
    data.append(FP16x16 { mag: 34110, sign: true });
    data.append(FP16x16 { mag: 50067, sign: false });
    data.append(FP16x16 { mag: 4048, sign: false });
    data.append(FP16x16 { mag: 19840, sign: false });
    TensorTrait::new(shape.span(), data.span())
}