use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 57550, sign: false });
    data.append(FP16x16 { mag: 313427, sign: true });
    data.append(FP16x16 { mag: 31559, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
