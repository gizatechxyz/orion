use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 74405172, sign: false });
    data.append(FP8x23 { mag: 9053304, sign: true });
    data.append(FP8x23 { mag: 26868073, sign: true });
    data.append(FP8x23 { mag: 74129735, sign: true });
    data.append(FP8x23 { mag: 21376636, sign: false });
    data.append(FP8x23 { mag: 22671396, sign: false });
    data.append(FP8x23 { mag: 127844813, sign: false });
    data.append(FP8x23 { mag: 132600746, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
