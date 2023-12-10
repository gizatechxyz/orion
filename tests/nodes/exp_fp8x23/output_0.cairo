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
    data.append(FP8x23 { mag: 2197802, sign: false });
    data.append(FP8x23 { mag: 15442702, sign: false });
    data.append(FP8x23 { mag: 7684813, sign: false });
    data.append(FP8x23 { mag: 855682, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
