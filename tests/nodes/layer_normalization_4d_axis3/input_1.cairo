use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 6200252, sign: false });
    data.append(FP8x23 { mag: 4645727, sign: true });
    data.append(FP8x23 { mag: 18131524, sign: false });
    data.append(FP8x23 { mag: 9670945, sign: true });
    data.append(FP8x23 { mag: 4399430, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
