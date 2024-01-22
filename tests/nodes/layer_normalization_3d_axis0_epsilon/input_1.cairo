use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 41116, sign: true });
    data.append(FP16x16 { mag: 33345, sign: false });
    data.append(FP16x16 { mag: 16035, sign: true });
    data.append(FP16x16 { mag: 66324, sign: true });
    data.append(FP16x16 { mag: 65521, sign: false });
    data.append(FP16x16 { mag: 44567, sign: false });
    data.append(FP16x16 { mag: 12343, sign: true });
    data.append(FP16x16 { mag: 22182, sign: false });
    data.append(FP16x16 { mag: 68006, sign: false });
    data.append(FP16x16 { mag: 18564, sign: true });
    data.append(FP16x16 { mag: 159060, sign: true });
    data.append(FP16x16 { mag: 41449, sign: false });
    data.append(FP16x16 { mag: 21502, sign: true });
    data.append(FP16x16 { mag: 23583, sign: true });
    data.append(FP16x16 { mag: 10004, sign: true });
    data.append(FP16x16 { mag: 111077, sign: true });
    data.append(FP16x16 { mag: 51023, sign: true });
    data.append(FP16x16 { mag: 27845, sign: false });
    data.append(FP16x16 { mag: 548, sign: false });
    data.append(FP16x16 { mag: 51940, sign: true });
    data.append(FP16x16 { mag: 4135, sign: false });
    data.append(FP16x16 { mag: 77167, sign: false });
    data.append(FP16x16 { mag: 27642, sign: true });
    data.append(FP16x16 { mag: 50063, sign: true });
    data.append(FP16x16 { mag: 60555, sign: true });
    data.append(FP16x16 { mag: 34923, sign: true });
    data.append(FP16x16 { mag: 66185, sign: false });
    data.append(FP16x16 { mag: 1845, sign: true });
    data.append(FP16x16 { mag: 63811, sign: true });
    data.append(FP16x16 { mag: 30524, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
