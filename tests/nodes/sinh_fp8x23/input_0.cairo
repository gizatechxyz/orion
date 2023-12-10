use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 20711629, sign: true });
    data.append(FP8x23 { mag: 18228194, sign: false });
    data.append(FP8x23 { mag: 15655651, sign: true });
    data.append(FP8x23 { mag: 13406669, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
