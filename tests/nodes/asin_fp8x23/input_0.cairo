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
    data.append(FP8x23 { mag: 5018963, sign: true });
    data.append(FP8x23 { mag: 1475178, sign: false });
    data.append(FP8x23 { mag: 4332453, sign: false });
    data.append(FP8x23 { mag: 3833460, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
