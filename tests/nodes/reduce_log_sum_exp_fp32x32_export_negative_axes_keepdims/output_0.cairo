use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP32x32Tensor;
use orion::numbers::{FixedTrait, FP32x32};

fn output_0() -> Tensor<FP32x32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP32x32 { mag: 38734073664, sign: false });
    data.append(FP32x32 { mag: 43029040960, sign: false });
    data.append(FP32x32 { mag: 47324008256, sign: false });
    data.append(FP32x32 { mag: 51618975552, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
