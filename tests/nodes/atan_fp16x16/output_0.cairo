use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 102063, sign: false });
    data.append(FP16x16 { mag: 89459, sign: true });
    data.append(FP16x16 { mag: 57320, sign: true });
    data.append(FP16x16 { mag: 94484, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
