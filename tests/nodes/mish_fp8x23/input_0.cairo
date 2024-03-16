use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 29330286, sign: true });
    data.append(FP8x23 { mag: 29576280, sign: false });
    data.append(FP8x23 { mag: 605854, sign: false });
    data.append(FP8x23 { mag: 26167402, sign: false });
    data.append(FP8x23 { mag: 24733382, sign: false });
    data.append(FP8x23 { mag: 5248967, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
