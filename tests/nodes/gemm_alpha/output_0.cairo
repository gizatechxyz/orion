use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 48876, sign: false });
    data.append(FP16x16 { mag: 38402, sign: false });
    data.append(FP16x16 { mag: 61940, sign: false });
    data.append(FP16x16 { mag: 42709, sign: false });
    data.append(FP16x16 { mag: 27783, sign: false });
    data.append(FP16x16 { mag: 40822, sign: false });
    data.append(FP16x16 { mag: 54034, sign: false });
    data.append(FP16x16 { mag: 39483, sign: false });
    data.append(FP16x16 { mag: 40179, sign: false });
    data.append(FP16x16 { mag: 20753, sign: false });
    data.append(FP16x16 { mag: 49237, sign: false });
    data.append(FP16x16 { mag: 36865, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
