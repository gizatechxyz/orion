use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorSub};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 13456914, sign: false });
    data.append(FP8x23 { mag: 15956279, sign: false });
    data.append(FP8x23 { mag: 16462601, sign: false });
    data.append(FP8x23 { mag: 15866344, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
