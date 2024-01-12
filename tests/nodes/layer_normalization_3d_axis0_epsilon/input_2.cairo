use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 12782, sign: true });
    data.append(FP16x16 { mag: 34829, sign: true });
    data.append(FP16x16 { mag: 85769, sign: true });
    data.append(FP16x16 { mag: 76891, sign: true });
    data.append(FP16x16 { mag: 52049, sign: true });
    data.append(FP16x16 { mag: 129289, sign: true });
    data.append(FP16x16 { mag: 92309, sign: true });
    data.append(FP16x16 { mag: 48090, sign: true });
    data.append(FP16x16 { mag: 1390, sign: false });
    data.append(FP16x16 { mag: 10093, sign: true });
    data.append(FP16x16 { mag: 6373, sign: true });
    data.append(FP16x16 { mag: 91002, sign: true });
    data.append(FP16x16 { mag: 9698, sign: false });
    data.append(FP16x16 { mag: 103992, sign: true });
    data.append(FP16x16 { mag: 26897, sign: true });
    data.append(FP16x16 { mag: 67478, sign: true });
    data.append(FP16x16 { mag: 5546, sign: false });
    data.append(FP16x16 { mag: 55870, sign: true });
    data.append(FP16x16 { mag: 35113, sign: true });
    data.append(FP16x16 { mag: 267167, sign: true });
    data.append(FP16x16 { mag: 51438, sign: true });
    data.append(FP16x16 { mag: 13667, sign: false });
    data.append(FP16x16 { mag: 17845, sign: false });
    data.append(FP16x16 { mag: 92263, sign: false });
    data.append(FP16x16 { mag: 114550, sign: true });
    data.append(FP16x16 { mag: 31510, sign: false });
    data.append(FP16x16 { mag: 24263, sign: true });
    data.append(FP16x16 { mag: 68737, sign: true });
    data.append(FP16x16 { mag: 61297, sign: true });
    data.append(FP16x16 { mag: 33386, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
