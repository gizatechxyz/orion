use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorDiv};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 429406, sign: false });
    data.append(FP16x16 { mag: 288523, sign: false });
    data.append(FP16x16 { mag: 414929, sign: false });
    data.append(FP16x16 { mag: 259811, sign: true });
    data.append(FP16x16 { mag: 394274, sign: false });
    data.append(FP16x16 { mag: 314398, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
