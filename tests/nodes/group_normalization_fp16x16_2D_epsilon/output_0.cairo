use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 61699, sign: false });
    data.append(FP16x16 { mag: 13438, sign: true });
    data.append(FP16x16 { mag: 14123, sign: false });
    data.append(FP16x16 { mag: 33294, sign: true });
    data.append(FP16x16 { mag: 101492, sign: false });
    data.append(FP16x16 { mag: 39773, sign: true });
    data.append(FP16x16 { mag: 53052, sign: true });
    data.append(FP16x16 { mag: 71859, sign: true });
    data.append(FP16x16 { mag: 104343, sign: false });
    data.append(FP16x16 { mag: 41661, sign: true });
    data.append(FP16x16 { mag: 3560, sign: false });
    data.append(FP16x16 { mag: 39358, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
