use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 38686, sign: false });
    data.append(FP16x16 { mag: 42453, sign: false });
    data.append(FP16x16 { mag: 35391, sign: false });
    data.append(FP16x16 { mag: 39825, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
