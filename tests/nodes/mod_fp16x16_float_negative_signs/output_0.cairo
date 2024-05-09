use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 125795, sign: true });
    data.append(FP16x16 { mag: 124957, sign: true });
    data.append(FP16x16 { mag: 116733, sign: true });
    data.append(FP16x16 { mag: 15390, sign: true });
    data.append(FP16x16 { mag: 91957, sign: true });
    data.append(FP16x16 { mag: 21058, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
