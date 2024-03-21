use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 96, sign: true });
    data.append(FP16x16 { mag: 43626, sign: false });
    data.append(FP16x16 { mag: 87593, sign: true });
    data.append(FP16x16 { mag: 51345, sign: true });
    data.append(FP16x16 { mag: 24632, sign: true });
    data.append(FP16x16 { mag: 35132, sign: false });
    data.append(FP16x16 { mag: 50661, sign: true });
    data.append(FP16x16 { mag: 26029, sign: false });
    data.append(FP16x16 { mag: 127274, sign: false });
    data.append(FP16x16 { mag: 10860, sign: true });
    data.append(FP16x16 { mag: 23624, sign: false });
    data.append(FP16x16 { mag: 60224, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
