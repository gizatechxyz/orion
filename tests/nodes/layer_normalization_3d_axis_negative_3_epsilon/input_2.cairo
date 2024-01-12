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
    data.append(FP16x16 { mag: 6143, sign: false });
    data.append(FP16x16 { mag: 4674, sign: false });
    data.append(FP16x16 { mag: 48051, sign: true });
    data.append(FP16x16 { mag: 18813, sign: false });
    data.append(FP16x16 { mag: 46995, sign: false });
    data.append(FP16x16 { mag: 20870, sign: true });
    data.append(FP16x16 { mag: 56843, sign: false });
    data.append(FP16x16 { mag: 81615, sign: false });
    data.append(FP16x16 { mag: 92340, sign: false });
    data.append(FP16x16 { mag: 84516, sign: false });
    data.append(FP16x16 { mag: 82019, sign: false });
    data.append(FP16x16 { mag: 51674, sign: false });
    data.append(FP16x16 { mag: 52303, sign: false });
    data.append(FP16x16 { mag: 7441, sign: true });
    data.append(FP16x16 { mag: 35138, sign: false });
    data.append(FP16x16 { mag: 78581, sign: false });
    data.append(FP16x16 { mag: 6660, sign: false });
    data.append(FP16x16 { mag: 137669, sign: true });
    data.append(FP16x16 { mag: 12790, sign: true });
    data.append(FP16x16 { mag: 144767, sign: false });
    data.append(FP16x16 { mag: 10893, sign: true });
    data.append(FP16x16 { mag: 26226, sign: true });
    data.append(FP16x16 { mag: 64470, sign: false });
    data.append(FP16x16 { mag: 22466, sign: false });
    data.append(FP16x16 { mag: 101996, sign: true });
    data.append(FP16x16 { mag: 46134, sign: true });
    data.append(FP16x16 { mag: 81851, sign: true });
    data.append(FP16x16 { mag: 176946, sign: false });
    data.append(FP16x16 { mag: 6446, sign: true });
    data.append(FP16x16 { mag: 77193, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
