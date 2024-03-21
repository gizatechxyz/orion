use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 7721, sign: true });
    data.append(FP16x16 { mag: 108326, sign: false });
    data.append(FP16x16 { mag: 126165, sign: true });
    data.append(FP16x16 { mag: 115219, sign: true });
    data.append(FP16x16 { mag: 149441, sign: true });
    data.append(FP16x16 { mag: 73390, sign: false });
    data.append(FP16x16 { mag: 126909, sign: true });
    data.append(FP16x16 { mag: 94376, sign: true });
    data.append(FP16x16 { mag: 128810, sign: false });
    data.append(FP16x16 { mag: 228266, sign: true });
    data.append(FP16x16 { mag: 114693, sign: true });
    data.append(FP16x16 { mag: 103929, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
