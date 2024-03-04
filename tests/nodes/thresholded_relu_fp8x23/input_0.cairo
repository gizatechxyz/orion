use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 11927593, sign: false });
    data.append(FP8x23 { mag: 2802160, sign: true });
    data.append(FP8x23 { mag: 23263089, sign: true });
    data.append(FP8x23 { mag: 38955682, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
