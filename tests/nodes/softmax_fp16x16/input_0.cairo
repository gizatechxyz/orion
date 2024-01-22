use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 93078, sign: true });
    data.append(FP16x16 { mag: 53025, sign: false });
    data.append(FP16x16 { mag: 183455, sign: true });
    data.append(FP16x16 { mag: 61044, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
