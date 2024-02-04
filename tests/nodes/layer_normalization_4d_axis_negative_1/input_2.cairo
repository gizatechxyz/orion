use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_2() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 14856029, sign: true });
    data.append(FP8x23 { mag: 11603919, sign: false });
    data.append(FP8x23 { mag: 2672200, sign: false });
    data.append(FP8x23 { mag: 513076, sign: false });
    data.append(FP8x23 { mag: 16056013, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
