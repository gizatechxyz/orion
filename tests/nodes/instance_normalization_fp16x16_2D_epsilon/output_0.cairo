use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 52227, sign: true });
    data.append(FP16x16 { mag: 93793, sign: true });
    data.append(FP16x16 { mag: 58000, sign: false });
    data.append(FP16x16 { mag: 100075, sign: false });
    data.append(FP16x16 { mag: 31814, sign: false });
    data.append(FP16x16 { mag: 52227, sign: true });
    data.append(FP16x16 { mag: 93793, sign: true });
    data.append(FP16x16 { mag: 58000, sign: false });
    data.append(FP16x16 { mag: 100075, sign: false });
    data.append(FP16x16 { mag: 31814, sign: false });
    data.append(FP16x16 { mag: 52227, sign: true });
    data.append(FP16x16 { mag: 93793, sign: true });
    data.append(FP16x16 { mag: 58000, sign: false });
    data.append(FP16x16 { mag: 100075, sign: false });
    data.append(FP16x16 { mag: 31814, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
