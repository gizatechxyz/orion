use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(1);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 13077381, sign: false });
    data.append(FP8x23 { mag: 414936, sign: true });
    data.append(FP8x23 { mag: 5248434, sign: true });
    data.append(FP8x23 { mag: 1062998, sign: true });
    data.append(FP8x23 { mag: 5055835, sign: true });
    data.append(FP8x23 { mag: 3133557, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
