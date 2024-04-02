use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2529683, sign: false });
    data.append(FP8x23 { mag: 1862726, sign: true });
    data.append(FP8x23 { mag: 7982227, sign: false });
    data.append(FP8x23 { mag: 1178555, sign: true });
    data.append(FP8x23 { mag: 9354192, sign: false });
    data.append(FP8x23 { mag: 10163695, sign: false });
    data.append(FP8x23 { mag: 6416861, sign: true });
    data.append(FP8x23 { mag: 7371997, sign: false });
    data.append(FP8x23 { mag: 5277977, sign: false });
    data.append(FP8x23 { mag: 13762680, sign: false });
    data.append(FP8x23 { mag: 14115532, sign: true });
    data.append(FP8x23 { mag: 1495824, sign: false });
    data.append(FP8x23 { mag: 3672804, sign: true });
    data.append(FP8x23 { mag: 5933537, sign: true });
    data.append(FP8x23 { mag: 9957890, sign: false });
    data.append(FP8x23 { mag: 17366184, sign: false });
    data.append(FP8x23 { mag: 13874303, sign: true });
    data.append(FP8x23 { mag: 3283032, sign: true });
    data.append(FP8x23 { mag: 11736718, sign: true });
    data.append(FP8x23 { mag: 769624, sign: true });
    data.append(FP8x23 { mag: 3451705, sign: true });
    data.append(FP8x23 { mag: 11553172, sign: true });
    data.append(FP8x23 { mag: 5600686, sign: true });
    data.append(FP8x23 { mag: 19169720, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
