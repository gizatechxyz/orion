use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 1396, sign: true });
    data.append(FP16x16 { mag: 91738, sign: false });
    data.append(FP16x16 { mag: 22354, sign: false });
    data.append(FP16x16 { mag: 40925, sign: true });
    data.append(FP16x16 { mag: 70890, sign: false });
    data.append(FP16x16 { mag: 64955, sign: false });
    data.append(FP16x16 { mag: 25324, sign: false });
    data.append(FP16x16 { mag: 25230, sign: false });
    data.append(FP16x16 { mag: 142906, sign: true });
    data.append(FP16x16 { mag: 68962, sign: true });
    data.append(FP16x16 { mag: 20940, sign: false });
    data.append(FP16x16 { mag: 107327, sign: true });
    data.append(FP16x16 { mag: 52698, sign: false });
    data.append(FP16x16 { mag: 17542, sign: true });
    data.append(FP16x16 { mag: 125557, sign: false });
    data.append(FP16x16 { mag: 3282, sign: false });
    data.append(FP16x16 { mag: 32752, sign: false });
    data.append(FP16x16 { mag: 38674, sign: false });
    data.append(FP16x16 { mag: 16699, sign: false });
    data.append(FP16x16 { mag: 20489, sign: false });
    data.append(FP16x16 { mag: 24738, sign: true });
    data.append(FP16x16 { mag: 21702, sign: false });
    data.append(FP16x16 { mag: 32990, sign: true });
    data.append(FP16x16 { mag: 73557, sign: false });
    data.append(FP16x16 { mag: 24845, sign: false });
    data.append(FP16x16 { mag: 48972, sign: false });
    data.append(FP16x16 { mag: 23684, sign: false });
    data.append(FP16x16 { mag: 168361, sign: false });
    data.append(FP16x16 { mag: 32800, sign: false });
    data.append(FP16x16 { mag: 66808, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
