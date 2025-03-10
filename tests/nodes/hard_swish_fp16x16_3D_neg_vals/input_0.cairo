use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 270984, sign: true });
    data.append(FP16x16 { mag: 211515, sign: true });
    data.append(FP16x16 { mag: 315936, sign: true });
    data.append(FP16x16 { mag: 218990, sign: true });
    data.append(FP16x16 { mag: 360176, sign: true });
    data.append(FP16x16 { mag: 247344, sign: true });
    data.append(FP16x16 { mag: 314131, sign: true });
    data.append(FP16x16 { mag: 268485, sign: true });
    data.append(FP16x16 { mag: 212524, sign: true });
    data.append(FP16x16 { mag: 219700, sign: true });
    data.append(FP16x16 { mag: 263508, sign: true });
    data.append(FP16x16 { mag: 291065, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
