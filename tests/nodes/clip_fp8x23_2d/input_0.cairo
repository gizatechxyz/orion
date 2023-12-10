use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 385875968, sign: true });
    data.append(FP8x23 { mag: 553648128, sign: false });
    data.append(FP8x23 { mag: 92274688, sign: true });
    data.append(FP8x23 { mag: 167772160, sign: true });
    data.append(FP8x23 { mag: 612368384, sign: false });
    data.append(FP8x23 { mag: 637534208, sign: false });
    data.append(FP8x23 { mag: 595591168, sign: true });
    data.append(FP8x23 { mag: 377487360, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
