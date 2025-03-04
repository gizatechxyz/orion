use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 42626, sign: false });
    data.append(FP16x16 { mag: 47228, sign: true });
    data.append(FP16x16 { mag: 20414, sign: true });
    data.append(FP16x16 { mag: 11334, sign: false });
    data.append(FP16x16 { mag: 99585, sign: false });
    data.append(FP16x16 { mag: 22597, sign: true });
    data.append(FP16x16 { mag: 68252, sign: true });
    data.append(FP16x16 { mag: 48845, sign: false });
    data.append(FP16x16 { mag: 10607, sign: false });
    data.append(FP16x16 { mag: 18093, sign: true });
    data.append(FP16x16 { mag: 2604, sign: false });
    data.append(FP16x16 { mag: 3834, sign: false });
    data.append(FP16x16 { mag: 94567, sign: false });
    data.append(FP16x16 { mag: 62584, sign: true });
    data.append(FP16x16 { mag: 122455, sign: false });
    data.append(FP16x16 { mag: 54601, sign: true });
    data.append(FP16x16 { mag: 30382, sign: false });
    data.append(FP16x16 { mag: 65383, sign: true });
    data.append(FP16x16 { mag: 161153, sign: true });
    data.append(FP16x16 { mag: 43629, sign: true });
    data.append(FP16x16 { mag: 51182, sign: false });
    data.append(FP16x16 { mag: 63325, sign: true });
    data.append(FP16x16 { mag: 15254, sign: false });
    data.append(FP16x16 { mag: 36705, sign: false });
    data.append(FP16x16 { mag: 47463, sign: true });
    data.append(FP16x16 { mag: 60670, sign: true });
    data.append(FP16x16 { mag: 17912, sign: false });
    data.append(FP16x16 { mag: 135318, sign: true });
    data.append(FP16x16 { mag: 27600, sign: true });
    data.append(FP16x16 { mag: 10328, sign: true });
    data.append(FP16x16 { mag: 18190, sign: true });
    data.append(FP16x16 { mag: 27637, sign: true });
    data.append(FP16x16 { mag: 1906, sign: false });
    data.append(FP16x16 { mag: 2999, sign: true });
    data.append(FP16x16 { mag: 21600, sign: false });
    data.append(FP16x16 { mag: 100427, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
