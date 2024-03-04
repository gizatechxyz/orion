use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 12195, sign: false });
    data.append(FP16x16 { mag: 25593, sign: false });
    data.append(FP16x16 { mag: 14094, sign: true });
    data.append(FP16x16 { mag: 99626, sign: true });
    data.append(FP16x16 { mag: 48676, sign: true });
    data.append(FP16x16 { mag: 58459, sign: false });
    data.append(FP16x16 { mag: 96699, sign: true });
    data.append(FP16x16 { mag: 14935, sign: true });
    data.append(FP16x16 { mag: 2362, sign: true });
    data.append(FP16x16 { mag: 150235, sign: false });
    data.append(FP16x16 { mag: 65730, sign: true });
    data.append(FP16x16 { mag: 56267, sign: false });
    data.append(FP16x16 { mag: 83617, sign: true });
    data.append(FP16x16 { mag: 34940, sign: false });
    data.append(FP16x16 { mag: 14826, sign: false });
    data.append(FP16x16 { mag: 67759, sign: true });
    data.append(FP16x16 { mag: 88099, sign: true });
    data.append(FP16x16 { mag: 103290, sign: true });
    data.append(FP16x16 { mag: 50684, sign: true });
    data.append(FP16x16 { mag: 29161, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
