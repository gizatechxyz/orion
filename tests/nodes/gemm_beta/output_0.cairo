use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 151333, sign: false });
    data.append(FP16x16 { mag: 112550, sign: false });
    data.append(FP16x16 { mag: 112879, sign: false });
    data.append(FP16x16 { mag: 109391, sign: false });
    data.append(FP16x16 { mag: 153762, sign: false });
    data.append(FP16x16 { mag: 129994, sign: false });
    data.append(FP16x16 { mag: 134574, sign: false });
    data.append(FP16x16 { mag: 128355, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
