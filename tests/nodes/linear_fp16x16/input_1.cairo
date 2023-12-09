use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 316249, sign: true });
    data.append(FP16x16 { mag: 223592, sign: true });
    data.append(FP16x16 { mag: 238282, sign: true });
    data.append(FP16x16 { mag: 452809, sign: false });
    data.append(FP16x16 { mag: 234567, sign: false });
    data.append(FP16x16 { mag: 135020, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
