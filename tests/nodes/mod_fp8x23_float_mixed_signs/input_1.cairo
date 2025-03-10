use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 9309842, sign: false });
    data.append(FP8x23 { mag: 80087864, sign: false });
    data.append(FP8x23 { mag: 32493216, sign: true });
    data.append(FP8x23 { mag: 46015200, sign: false });
    data.append(FP8x23 { mag: 6887932, sign: false });
    data.append(FP8x23 { mag: 72537160, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
