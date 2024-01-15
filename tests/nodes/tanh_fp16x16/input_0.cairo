use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 178363, sign: false });
    data.append(FP16x16 { mag: 92058, sign: true });
    data.append(FP16x16 { mag: 105941, sign: false });
    data.append(FP16x16 { mag: 33142, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
