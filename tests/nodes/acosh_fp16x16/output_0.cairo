use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 133952, sign: false });
    data.append(FP16x16 { mag: 138187, sign: false });
    data.append(FP16x16 { mag: 111495, sign: false });
    data.append(FP16x16 { mag: 146089, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
