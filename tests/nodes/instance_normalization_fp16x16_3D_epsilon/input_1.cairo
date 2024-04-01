use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 1448, sign: false });
    data.append(FP16x16 { mag: 32209, sign: false });
    data.append(FP16x16 { mag: 26762, sign: false });
    data.append(FP16x16 { mag: 15168, sign: false });
    data.append(FP16x16 { mag: 69183, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
