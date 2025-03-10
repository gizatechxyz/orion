use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 90709, sign: false });
    data.append(FP16x16 { mag: 137987, sign: false });
    data.append(FP16x16 { mag: 36103, sign: false });
    data.append(FP16x16 { mag: 9820, sign: false });
    data.append(FP16x16 { mag: 8390, sign: false });
    data.append(FP16x16 { mag: 3284, sign: false });
    data.append(FP16x16 { mag: 12034, sign: true });
    data.append(FP16x16 { mag: 20542, sign: true });
    data.append(FP16x16 { mag: 117346, sign: false });
    data.append(FP16x16 { mag: 5159, sign: false });
    data.append(FP16x16 { mag: 101287, sign: false });
    data.append(FP16x16 { mag: 148819, sign: false });
    data.append(FP16x16 { mag: 9123, sign: true });
    data.append(FP16x16 { mag: 23704, sign: true });
    data.append(FP16x16 { mag: 4727, sign: false });
    data.append(FP16x16 { mag: 38931, sign: false });
    data.append(FP16x16 { mag: 146176, sign: false });
    data.append(FP16x16 { mag: 14168, sign: true });
    data.append(FP16x16 { mag: 6969, sign: false });
    data.append(FP16x16 { mag: 22342, sign: true });
    data.append(FP16x16 { mag: 2035, sign: true });
    data.append(FP16x16 { mag: 83786, sign: false });
    data.append(FP16x16 { mag: 140076, sign: false });
    data.append(FP16x16 { mag: 75359, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
