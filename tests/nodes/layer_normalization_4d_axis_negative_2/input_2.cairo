use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_2() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 8789918, sign: true });
    data.append(FP8x23 { mag: 14437992, sign: false });
    data.append(FP8x23 { mag: 86356, sign: true });
    data.append(FP8x23 { mag: 3036480, sign: true });
    data.append(FP8x23 { mag: 6736255, sign: true });
    data.append(FP8x23 { mag: 5972939, sign: false });
    data.append(FP8x23 { mag: 2010263, sign: false });
    data.append(FP8x23 { mag: 6592165, sign: false });
    data.append(FP8x23 { mag: 764864, sign: false });
    data.append(FP8x23 { mag: 6630163, sign: true });
    data.append(FP8x23 { mag: 13792561, sign: true });
    data.append(FP8x23 { mag: 7577680, sign: true });
    data.append(FP8x23 { mag: 12301127, sign: false });
    data.append(FP8x23 { mag: 2468466, sign: false });
    data.append(FP8x23 { mag: 766822, sign: true });
    data.append(FP8x23 { mag: 9420083, sign: false });
    data.append(FP8x23 { mag: 2868397, sign: false });
    data.append(FP8x23 { mag: 9292738, sign: true });
    data.append(FP8x23 { mag: 6785176, sign: false });
    data.append(FP8x23 { mag: 18482062, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
