use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 114609, sign: false });
    data.append(FP16x16 { mag: 154525, sign: false });
    data.append(FP16x16 { mag: 56163, sign: false });
    data.append(FP16x16 { mag: 17994, sign: false });
    data.append(FP16x16 { mag: 15551, sign: false });
    data.append(FP16x16 { mag: 6362, sign: false });
    data.append(FP16x16 { mag: 168529, sign: true });
    data.append(FP16x16 { mag: 58478, sign: true });
    data.append(FP16x16 { mag: 137928, sign: false });
    data.append(FP16x16 { mag: 9827, sign: false });
    data.append(FP16x16 { mag: 124162, sign: false });
    data.append(FP16x16 { mag: 162812, sign: false });
    data.append(FP16x16 { mag: 20354, sign: true });
    data.append(FP16x16 { mag: 79796, sign: true });
    data.append(FP16x16 { mag: 9039, sign: false });
    data.append(FP16x16 { mag: 59721, sign: false });
    data.append(FP16x16 { mag: 160814, sign: false });
    data.append(FP16x16 { mag: 34332, sign: true });
    data.append(FP16x16 { mag: 13070, sign: false });
    data.append(FP16x16 { mag: 127938, sign: true });
    data.append(FP16x16 { mag: 4158, sign: true });
    data.append(FP16x16 { mag: 108117, sign: false });
    data.append(FP16x16 { mag: 156144, sign: false });
    data.append(FP16x16 { mag: 99928, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
