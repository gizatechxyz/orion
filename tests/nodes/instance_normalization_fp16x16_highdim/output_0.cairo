use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(2);
    shape.append(3);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 213497, sign: false });
    data.append(FP16x16 { mag: 21331, sign: false });
    data.append(FP16x16 { mag: 88830, sign: false });
    data.append(FP16x16 { mag: 8610, sign: true });
    data.append(FP16x16 { mag: 69563, sign: false });
    data.append(FP16x16 { mag: 22649, sign: false });
    data.append(FP16x16 { mag: 222453, sign: true });
    data.append(FP16x16 { mag: 135385, sign: true });
    data.append(FP16x16 { mag: 55177, sign: true });
    data.append(FP16x16 { mag: 16659, sign: true });
    data.append(FP16x16 { mag: 11020, sign: true });
    data.append(FP16x16 { mag: 110453, sign: false });
    data.append(FP16x16 { mag: 83090, sign: false });
    data.append(FP16x16 { mag: 33861, sign: true });
    data.append(FP16x16 { mag: 9586, sign: true });
    data.append(FP16x16 { mag: 35327, sign: true });
    data.append(FP16x16 { mag: 51739, sign: true });
    data.append(FP16x16 { mag: 56788, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
