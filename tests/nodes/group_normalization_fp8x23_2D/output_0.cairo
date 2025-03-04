use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 8743698, sign: false });
    data.append(FP8x23 { mag: 1099025, sign: true });
    data.append(FP8x23 { mag: 10774666, sign: false });
    data.append(FP8x23 { mag: 2737037, sign: true });
    data.append(FP8x23 { mag: 8745600, sign: false });
    data.append(FP8x23 { mag: 1091064, sign: true });
    data.append(FP8x23 { mag: 14882376, sign: true });
    data.append(FP8x23 { mag: 1039074, sign: true });
    data.append(FP8x23 { mag: 4909308, sign: false });
    data.append(FP8x23 { mag: 17181212, sign: true });
    data.append(FP8x23 { mag: 10775493, sign: false });
    data.append(FP8x23 { mag: 2737091, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
