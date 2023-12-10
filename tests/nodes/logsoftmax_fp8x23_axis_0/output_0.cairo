use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2191868, sign: true });
    data.append(FP8x23 { mag: 10484172, sign: true });
    data.append(FP8x23 { mag: 12330606, sign: true });
    data.append(FP8x23 { mag: 2832460, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
