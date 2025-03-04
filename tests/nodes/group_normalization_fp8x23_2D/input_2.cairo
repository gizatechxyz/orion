use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_2() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 6827246, sign: false });
    data.append(FP8x23 { mag: 9137015, sign: true });
    data.append(FP8x23 { mag: 2053494, sign: true });
    data.append(FP8x23 { mag: 1888079, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
