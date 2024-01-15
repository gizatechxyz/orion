use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 30, sign: false });
    data.append(FP8x23 { mag: 33, sign: false });
    data.append(FP8x23 { mag: 36, sign: false });
    data.append(FP8x23 { mag: 39, sign: false });
    data.append(FP8x23 { mag: 42, sign: false });
    data.append(FP8x23 { mag: 45, sign: false });
    data.append(FP8x23 { mag: 48, sign: false });
    data.append(FP8x23 { mag: 51, sign: false });
    data.append(FP8x23 { mag: 54, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
