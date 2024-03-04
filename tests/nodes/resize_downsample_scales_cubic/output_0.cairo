use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 96416, sign: false });
    data.append(FP16x16 { mag: 182272, sign: false });
    data.append(FP16x16 { mag: 267552, sign: false });
    data.append(FP16x16 { mag: 439840, sign: false });
    data.append(FP16x16 { mag: 525696, sign: false });
    data.append(FP16x16 { mag: 610976, sign: false });
    data.append(FP16x16 { mag: 780960, sign: false });
    data.append(FP16x16 { mag: 866816, sign: false });
    data.append(FP16x16 { mag: 952096, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
