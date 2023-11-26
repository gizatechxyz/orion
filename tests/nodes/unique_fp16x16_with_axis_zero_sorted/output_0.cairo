use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 132608, sign: false });
    data.append(FP16x16 { mag: 9192, sign: true });
    data.append(FP16x16 { mag: 76160, sign: false });
    data.append(FP16x16 { mag: 59712, sign: true });
    data.append(FP16x16 { mag: 56256, sign: true });
    data.append(FP16x16 { mag: 161024, sign: false });
    data.append(FP16x16 { mag: 24224, sign: true });
    data.append(FP16x16 { mag: 9280, sign: false });
    data.append(FP16x16 { mag: 38336, sign: true });
    data.append(FP16x16 { mag: 73792, sign: true });
    data.append(FP16x16 { mag: 23888, sign: true });
    data.append(FP16x16 { mag: 25152, sign: false });
    data.append(FP16x16 { mag: 63776, sign: true });
    data.append(FP16x16 { mag: 163840, sign: true });
    data.append(FP16x16 { mag: 4688, sign: true });
    data.append(FP16x16 { mag: 151808, sign: true });
    data.append(FP16x16 { mag: 151680, sign: false });
    data.append(FP16x16 { mag: 44704, sign: false });
    data.append(FP16x16 { mag: 190208, sign: false });
    data.append(FP16x16 { mag: 144000, sign: true });
    data.append(FP16x16 { mag: 89472, sign: true });
    data.append(FP16x16 { mag: 47520, sign: true });
    data.append(FP16x16 { mag: 143872, sign: true });
    data.append(FP16x16 { mag: 11672, sign: false });
    data.append(FP16x16 { mag: 163840, sign: true });
    data.append(FP16x16 { mag: 4604, sign: false });
    data.append(FP16x16 { mag: 173696, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
