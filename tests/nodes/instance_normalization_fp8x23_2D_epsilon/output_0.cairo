use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 14389910, sign: false });
    data.append(FP8x23 { mag: 634668, sign: true });
    data.append(FP8x23 { mag: 21061910, sign: true });
    data.append(FP8x23 { mag: 52644, sign: true });
    data.append(FP8x23 { mag: 1324035, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
