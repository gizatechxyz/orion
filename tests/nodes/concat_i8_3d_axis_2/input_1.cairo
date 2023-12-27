use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 27, sign: false });
    data.append(FP8x23 { mag: 28, sign: false });
    data.append(FP8x23 { mag: 29, sign: false });
    data.append(FP8x23 { mag: 30, sign: false });
    data.append(FP8x23 { mag: 31, sign: false });
    data.append(FP8x23 { mag: 32, sign: false });
    data.append(FP8x23 { mag: 33, sign: false });
    data.append(FP8x23 { mag: 34, sign: false });
    data.append(FP8x23 { mag: 35, sign: false });
    data.append(FP8x23 { mag: 36, sign: false });
    data.append(FP8x23 { mag: 37, sign: false });
    data.append(FP8x23 { mag: 38, sign: false });
    data.append(FP8x23 { mag: 39, sign: false });
    data.append(FP8x23 { mag: 40, sign: false });
    data.append(FP8x23 { mag: 41, sign: false });
    data.append(FP8x23 { mag: 42, sign: false });
    data.append(FP8x23 { mag: 43, sign: false });
    data.append(FP8x23 { mag: 44, sign: false });
    data.append(FP8x23 { mag: 45, sign: false });
    data.append(FP8x23 { mag: 46, sign: false });
    data.append(FP8x23 { mag: 47, sign: false });
    data.append(FP8x23 { mag: 48, sign: false });
    data.append(FP8x23 { mag: 49, sign: false });
    data.append(FP8x23 { mag: 50, sign: false });
    data.append(FP8x23 { mag: 51, sign: false });
    data.append(FP8x23 { mag: 52, sign: false });
    data.append(FP8x23 { mag: 53, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
