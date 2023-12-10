use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 21, sign: false });
    data.append(FP16x16 { mag: 22, sign: false });
    data.append(FP16x16 { mag: 24, sign: false });
    data.append(FP16x16 { mag: 25, sign: false });
    data.append(FP16x16 { mag: 27, sign: false });
    data.append(FP16x16 { mag: 28, sign: false });
    data.append(FP16x16 { mag: 30, sign: false });
    data.append(FP16x16 { mag: 32, sign: false });
    data.append(FP16x16 { mag: 33, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
