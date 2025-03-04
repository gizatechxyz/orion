use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 8736308, sign: true });
    data.append(FP8x23 { mag: 18148086, sign: false });
    data.append(FP8x23 { mag: 1801176, sign: false });
    data.append(FP8x23 { mag: 1956545, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
