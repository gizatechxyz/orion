use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(2);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 49587, sign: false });
    data.append(FP16x16 { mag: 53758, sign: false });
    data.append(FP16x16 { mag: 25226, sign: false });
    data.append(FP16x16 { mag: 35287, sign: false });
    data.append(FP16x16 { mag: 27302, sign: false });
    data.append(FP16x16 { mag: 155257, sign: true });
    data.append(FP16x16 { mag: 116915, sign: false });
    data.append(FP16x16 { mag: 55602, sign: false });
    data.append(FP16x16 { mag: 80322, sign: false });
    data.append(FP16x16 { mag: 49009, sign: true });
    data.append(FP16x16 { mag: 47876, sign: false });
    data.append(FP16x16 { mag: 41339, sign: false });
    data.append(FP16x16 { mag: 52300, sign: false });
    data.append(FP16x16 { mag: 22937, sign: false });
    data.append(FP16x16 { mag: 26709, sign: false });
    data.append(FP16x16 { mag: 30978, sign: true });
    data.append(FP16x16 { mag: 143639, sign: true });
    data.append(FP16x16 { mag: 59608, sign: false });
    data.append(FP16x16 { mag: 156608, sign: false });
    data.append(FP16x16 { mag: 6975, sign: false });
    data.append(FP16x16 { mag: 46855, sign: false });
    data.append(FP16x16 { mag: 56336, sign: false });
    data.append(FP16x16 { mag: 29455, sign: false });
    data.append(FP16x16 { mag: 32937, sign: false });
    data.append(FP16x16 { mag: 25578, sign: false });
    data.append(FP16x16 { mag: 6083, sign: true });
    data.append(FP16x16 { mag: 131044, sign: true });
    data.append(FP16x16 { mag: 123470, sign: false });
    data.append(FP16x16 { mag: 56430, sign: true });
    data.append(FP16x16 { mag: 118661, sign: false });
    data.append(FP16x16 { mag: 27906, sign: false });
    data.append(FP16x16 { mag: 35865, sign: false });
    data.append(FP16x16 { mag: 29196, sign: false });
    data.append(FP16x16 { mag: 38204, sign: false });
    data.append(FP16x16 { mag: 59990, sign: false });
    data.append(FP16x16 { mag: 74018, sign: false });
    data.append(FP16x16 { mag: 110398, sign: true });
    data.append(FP16x16 { mag: 51101, sign: true });
    data.append(FP16x16 { mag: 32435, sign: true });
    data.append(FP16x16 { mag: 168491, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
