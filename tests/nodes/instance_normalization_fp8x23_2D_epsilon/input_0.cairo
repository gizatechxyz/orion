use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1606779, sign: false });
    data.append(FP8x23 { mag: 19632498, sign: false });
    data.append(FP8x23 { mag: 524185, sign: true });
    data.append(FP8x23 { mag: 1011053, sign: false });
    data.append(FP8x23 { mag: 23798, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
