use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 38618, sign: false });
    data.append(FP16x16 { mag: 46835, sign: true });
    data.append(FP16x16 { mag: 183371, sign: true });
    data.append(FP16x16 { mag: 60520, sign: true });
    data.append(FP16x16 { mag: 134389, sign: false });
    data.append(FP16x16 { mag: 10773, sign: false });
    data.append(FP16x16 { mag: 111121, sign: true });
    data.append(FP16x16 { mag: 155827, sign: false });
    data.append(FP16x16 { mag: 50600, sign: true });
    data.append(FP16x16 { mag: 184242, sign: false });
    data.append(FP16x16 { mag: 134602, sign: true });
    data.append(FP16x16 { mag: 147290, sign: false });
    data.append(FP16x16 { mag: 117135, sign: true });
    data.append(FP16x16 { mag: 50535, sign: false });
    data.append(FP16x16 { mag: 134502, sign: false });
    data.append(FP16x16 { mag: 115730, sign: false });
    data.append(FP16x16 { mag: 92700, sign: false });
    data.append(FP16x16 { mag: 78606, sign: false });
    data.append(FP16x16 { mag: 175904, sign: false });
    data.append(FP16x16 { mag: 23346, sign: true });
    data.append(FP16x16 { mag: 149382, sign: true });
    data.append(FP16x16 { mag: 187601, sign: false });
    data.append(FP16x16 { mag: 180445, sign: true });
    data.append(FP16x16 { mag: 143605, sign: true });
    data.append(FP16x16 { mag: 108591, sign: true });
    data.append(FP16x16 { mag: 129062, sign: false });
    data.append(FP16x16 { mag: 180135, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
