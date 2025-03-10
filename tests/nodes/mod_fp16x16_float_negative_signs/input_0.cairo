use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 388210, sign: true });
    data.append(FP16x16 { mag: 124957, sign: true });
    data.append(FP16x16 { mag: 116733, sign: true });
    data.append(FP16x16 { mag: 378014, sign: true });
    data.append(FP16x16 { mag: 91957, sign: true });
    data.append(FP16x16 { mag: 271493, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
