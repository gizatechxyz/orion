use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 3053556, sign: true });
    data.append(FP8x23 { mag: 2883899, sign: true });
    data.append(FP8x23 { mag: 1998702, sign: true });
    data.append(FP8x23 { mag: 19354288, sign: false });
    data.append(FP8x23 { mag: 60271, sign: true });
    data.append(FP8x23 { mag: 6091439, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
