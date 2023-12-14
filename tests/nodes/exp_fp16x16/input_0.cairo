use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 194862, sign: false });
    data.append(FP16x16 { mag: 38141, sign: true });
    data.append(FP16x16 { mag: 110238, sign: false });
    data.append(FP16x16 { mag: 63859, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
