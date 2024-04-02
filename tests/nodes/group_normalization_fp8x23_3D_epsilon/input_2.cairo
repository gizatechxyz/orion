use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_2() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2013188, sign: false });
    data.append(FP8x23 { mag: 160550, sign: false });
    data.append(FP8x23 { mag: 5598850, sign: true });
    data.append(FP8x23 { mag: 6892397, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
