use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 19493, sign: true });
    data.append(FP16x16 { mag: 6259, sign: true });
    data.append(FP16x16 { mag: 35118, sign: true });
    data.append(FP16x16 { mag: 2823, sign: false });
    data.append(FP16x16 { mag: 34210, sign: false });
    data.append(FP16x16 { mag: 7012, sign: false });
    data.append(FP16x16 { mag: 52120, sign: true });
    data.append(FP16x16 { mag: 51293, sign: true });
    data.append(FP16x16 { mag: 23181, sign: false });
    data.append(FP16x16 { mag: 54682, sign: false });
    data.append(FP16x16 { mag: 34168, sign: false });
    data.append(FP16x16 { mag: 54896, sign: false });
    data.append(FP16x16 { mag: 94902, sign: false });
    data.append(FP16x16 { mag: 35974, sign: true });
    data.append(FP16x16 { mag: 96972, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
