use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 30641, sign: false });
    data.append(FP16x16 { mag: 53716, sign: false });
    data.append(FP16x16 { mag: 41, sign: false });
    data.append(FP16x16 { mag: 47279, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
