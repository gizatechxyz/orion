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
    data.append(FP16x16 { mag: 8634, sign: false });
    data.append(FP16x16 { mag: 28856, sign: false });
    data.append(FP16x16 { mag: 182071, sign: false });
    data.append(FP16x16 { mag: 3113, sign: false });
    data.append(FP16x16 { mag: 159782, sign: false });
    data.append(FP16x16 { mag: 74461, sign: true });
    data.append(FP16x16 { mag: 26075, sign: false });
    data.append(FP16x16 { mag: 125670, sign: false });
    data.append(FP16x16 { mag: 17113, sign: false });
    data.append(FP16x16 { mag: 87340, sign: false });
    data.append(FP16x16 { mag: 162564, sign: false });
    data.append(FP16x16 { mag: 24194, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
