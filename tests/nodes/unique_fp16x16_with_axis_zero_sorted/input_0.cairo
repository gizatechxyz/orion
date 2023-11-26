use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 132575, sign: false });
    data.append(FP16x16 { mag: 9190, sign: true });
    data.append(FP16x16 { mag: 76177, sign: false });
    data.append(FP16x16 { mag: 59707, sign: true });
    data.append(FP16x16 { mag: 56266, sign: true });
    data.append(FP16x16 { mag: 161016, sign: false });
    data.append(FP16x16 { mag: 24216, sign: true });
    data.append(FP16x16 { mag: 9276, sign: false });
    data.append(FP16x16 { mag: 38345, sign: true });
    data.append(FP16x16 { mag: 73795, sign: true });
    data.append(FP16x16 { mag: 23884, sign: true });
    data.append(FP16x16 { mag: 25152, sign: false });
    data.append(FP16x16 { mag: 63762, sign: true });
    data.append(FP16x16 { mag: 163887, sign: true });
    data.append(FP16x16 { mag: 4688, sign: true });
    data.append(FP16x16 { mag: 151787, sign: true });
    data.append(FP16x16 { mag: 151718, sign: false });
    data.append(FP16x16 { mag: 44692, sign: false });
    data.append(FP16x16 { mag: 190155, sign: false });
    data.append(FP16x16 { mag: 144034, sign: true });
    data.append(FP16x16 { mag: 89483, sign: true });
    data.append(FP16x16 { mag: 47530, sign: true });
    data.append(FP16x16 { mag: 143886, sign: true });
    data.append(FP16x16 { mag: 11671, sign: false });
    data.append(FP16x16 { mag: 163828, sign: true });
    data.append(FP16x16 { mag: 4602, sign: false });
    data.append(FP16x16 { mag: 173634, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
