use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 114840, sign: true });
    data.append(FP16x16 { mag: 90046, sign: true });
    data.append(FP16x16 { mag: 13419, sign: true });
    data.append(FP16x16 { mag: 67891, sign: false });
    data.append(FP16x16 { mag: 2473, sign: false });
    data.append(FP16x16 { mag: 53134, sign: true });
    data.append(FP16x16 { mag: 42214, sign: true });
    data.append(FP16x16 { mag: 62961, sign: false });
    data.append(FP16x16 { mag: 19822, sign: true });
    data.append(FP16x16 { mag: 263248, sign: false });
    data.append(FP16x16 { mag: 36043, sign: false });
    data.append(FP16x16 { mag: 39237, sign: false });
    data.append(FP16x16 { mag: 97766, sign: true });
    data.append(FP16x16 { mag: 47816, sign: false });
    data.append(FP16x16 { mag: 17567, sign: true });
    data.append(FP16x16 { mag: 26247, sign: true });
    data.append(FP16x16 { mag: 26852, sign: false });
    data.append(FP16x16 { mag: 46147, sign: true });
    data.append(FP16x16 { mag: 87501, sign: false });
    data.append(FP16x16 { mag: 18885, sign: false });
    data.append(FP16x16 { mag: 46581, sign: true });
    data.append(FP16x16 { mag: 51141, sign: true });
    data.append(FP16x16 { mag: 5208, sign: false });
    data.append(FP16x16 { mag: 19988, sign: true });
    data.append(FP16x16 { mag: 29072, sign: true });
    data.append(FP16x16 { mag: 16318, sign: false });
    data.append(FP16x16 { mag: 197324, sign: false });
    data.append(FP16x16 { mag: 79100, sign: true });
    data.append(FP16x16 { mag: 60801, sign: false });
    data.append(FP16x16 { mag: 16170, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
