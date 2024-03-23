use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 16439, sign: false });
    data.append(FP16x16 { mag: 7360, sign: false });
    data.append(FP16x16 { mag: 37088, sign: false });
    data.append(FP16x16 { mag: 9738, sign: true });
    data.append(FP16x16 { mag: 51560, sign: true });
    data.append(FP16x16 { mag: 141950, sign: true });
    data.append(FP16x16 { mag: 64768, sign: false });
    data.append(FP16x16 { mag: 136524, sign: true });
    data.append(FP16x16 { mag: 13107, sign: true });
    data.append(FP16x16 { mag: 8016, sign: false });
    data.append(FP16x16 { mag: 12003, sign: true });
    data.append(FP16x16 { mag: 44828, sign: true });
    data.append(FP16x16 { mag: 52186, sign: false });
    data.append(FP16x16 { mag: 28580, sign: false });
    data.append(FP16x16 { mag: 135126, sign: true });
    data.append(FP16x16 { mag: 88310, sign: true });
    data.append(FP16x16 { mag: 50552, sign: true });
    data.append(FP16x16 { mag: 49884, sign: false });
    data.append(FP16x16 { mag: 82183, sign: false });
    data.append(FP16x16 { mag: 44715, sign: false });
    data.append(FP16x16 { mag: 63228, sign: true });
    data.append(FP16x16 { mag: 77884, sign: true });
    data.append(FP16x16 { mag: 57346, sign: false });
    data.append(FP16x16 { mag: 161613, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
