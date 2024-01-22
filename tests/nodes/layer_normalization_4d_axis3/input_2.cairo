use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_2() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 14176189, sign: false });
    data.append(FP8x23 { mag: 13256247, sign: false });
    data.append(FP8x23 { mag: 7459632, sign: true });
    data.append(FP8x23 { mag: 2183266, sign: false });
    data.append(FP8x23 { mag: 3222205, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
