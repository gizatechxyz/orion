use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 31342, sign: true });
    data.append(FP16x16 { mag: 65832, sign: false });
    data.append(FP16x16 { mag: 21387, sign: true });
    data.append(FP16x16 { mag: 106508, sign: false });
    data.append(FP16x16 { mag: 13277, sign: false });
    data.append(FP16x16 { mag: 26174, sign: false });
    data.append(FP16x16 { mag: 34514, sign: false });
    data.append(FP16x16 { mag: 36889, sign: false });
    data.append(FP16x16 { mag: 16939, sign: true });
    data.append(FP16x16 { mag: 110741, sign: false });
    data.append(FP16x16 { mag: 33103, sign: true });
    data.append(FP16x16 { mag: 58912, sign: false });
    data.append(FP16x16 { mag: 26004, sign: false });
    data.append(FP16x16 { mag: 13289, sign: false });
    data.append(FP16x16 { mag: 35588, sign: false });
    data.append(FP16x16 { mag: 35974, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
