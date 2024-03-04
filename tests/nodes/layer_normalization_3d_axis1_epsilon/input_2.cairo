use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 37910, sign: true });
    data.append(FP16x16 { mag: 18888, sign: true });
    data.append(FP16x16 { mag: 42890, sign: false });
    data.append(FP16x16 { mag: 47165, sign: false });
    data.append(FP16x16 { mag: 63877, sign: true });
    data.append(FP16x16 { mag: 88993, sign: true });
    data.append(FP16x16 { mag: 41540, sign: true });
    data.append(FP16x16 { mag: 68862, sign: true });
    data.append(FP16x16 { mag: 21940, sign: false });
    data.append(FP16x16 { mag: 5420, sign: true });
    data.append(FP16x16 { mag: 101888, sign: true });
    data.append(FP16x16 { mag: 13856, sign: false });
    data.append(FP16x16 { mag: 55772, sign: false });
    data.append(FP16x16 { mag: 21988, sign: false });
    data.append(FP16x16 { mag: 63379, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
