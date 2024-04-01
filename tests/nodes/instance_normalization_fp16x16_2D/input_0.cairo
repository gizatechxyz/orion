use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 34675, sign: false });
    data.append(FP16x16 { mag: 7603, sign: true });
    data.append(FP16x16 { mag: 76758, sign: true });
    data.append(FP16x16 { mag: 69921, sign: true });
    data.append(FP16x16 { mag: 15866, sign: true });
    data.append(FP16x16 { mag: 108268, sign: false });
    data.append(FP16x16 { mag: 34420, sign: false });
    data.append(FP16x16 { mag: 70631, sign: false });
    data.append(FP16x16 { mag: 43834, sign: true });
    data.append(FP16x16 { mag: 23480, sign: false });
    data.append(FP16x16 { mag: 34940, sign: true });
    data.append(FP16x16 { mag: 120870, sign: false });
    data.append(FP16x16 { mag: 103977, sign: true });
    data.append(FP16x16 { mag: 7361, sign: true });
    data.append(FP16x16 { mag: 138907, sign: true });
    data.append(FP16x16 { mag: 3623, sign: false });
    data.append(FP16x16 { mag: 1419, sign: true });
    data.append(FP16x16 { mag: 35921, sign: true });
    data.append(FP16x16 { mag: 99392, sign: true });
    data.append(FP16x16 { mag: 89624, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
