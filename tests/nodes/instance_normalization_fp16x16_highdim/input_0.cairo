use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(2);
    shape.append(3);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 112724, sign: true });
    data.append(FP16x16 { mag: 25837, sign: false });
    data.append(FP16x16 { mag: 22833, sign: true });
    data.append(FP16x16 { mag: 47427, sign: false });
    data.append(FP16x16 { mag: 8940, sign: true });
    data.append(FP16x16 { mag: 24887, sign: false });
    data.append(FP16x16 { mag: 64545, sign: true });
    data.append(FP16x16 { mag: 14314, sign: true });
    data.append(FP16x16 { mag: 31958, sign: false });
    data.append(FP16x16 { mag: 54180, sign: false });
    data.append(FP16x16 { mag: 57433, sign: false });
    data.append(FP16x16 { mag: 127513, sign: false });
    data.append(FP16x16 { mag: 82422, sign: false });
    data.append(FP16x16 { mag: 35150, sign: true });
    data.append(FP16x16 { mag: 10746, sign: true });
    data.append(FP16x16 { mag: 36624, sign: true });
    data.append(FP16x16 { mag: 53123, sign: true });
    data.append(FP16x16 { mag: 55981, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
