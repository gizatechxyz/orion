use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 16960484, sign: false });
    data.append(FP8x23 { mag: 2604308, sign: true });
    data.append(FP8x23 { mag: 17884380, sign: true });
    data.append(FP8x23 { mag: 10159515, sign: false });
    data.append(FP8x23 { mag: 678914, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
