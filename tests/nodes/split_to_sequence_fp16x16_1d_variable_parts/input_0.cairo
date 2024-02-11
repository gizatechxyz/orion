use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 2752512, sign: true });
    data.append(FP16x16 { mag: 6750208, sign: false });
    data.append(FP16x16 { mag: 4849664, sign: true });
    data.append(FP16x16 { mag: 1966080, sign: false });
    data.append(FP16x16 { mag: 7733248, sign: false });
    data.append(FP16x16 { mag: 1835008, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
