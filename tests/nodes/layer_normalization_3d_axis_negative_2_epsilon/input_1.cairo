use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 76875, sign: true });
    data.append(FP16x16 { mag: 70595, sign: true });
    data.append(FP16x16 { mag: 48362, sign: true });
    data.append(FP16x16 { mag: 114023, sign: false });
    data.append(FP16x16 { mag: 68398, sign: true });
    data.append(FP16x16 { mag: 90609, sign: false });
    data.append(FP16x16 { mag: 11920, sign: false });
    data.append(FP16x16 { mag: 83372, sign: true });
    data.append(FP16x16 { mag: 131126, sign: false });
    data.append(FP16x16 { mag: 36226, sign: false });
    data.append(FP16x16 { mag: 15255, sign: true });
    data.append(FP16x16 { mag: 97966, sign: false });
    data.append(FP16x16 { mag: 156224, sign: false });
    data.append(FP16x16 { mag: 92550, sign: true });
    data.append(FP16x16 { mag: 120464, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
