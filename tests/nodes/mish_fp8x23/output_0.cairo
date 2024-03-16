use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 875391, sign: true });
    data.append(FP8x23 { mag: 29527976, sign: false });
    data.append(FP8x23 { mag: 377454, sign: false });
    data.append(FP8x23 { mag: 26073864, sign: false });
    data.append(FP8x23 { mag: 24610957, sign: false });
    data.append(FP8x23 { mag: 2120704, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
