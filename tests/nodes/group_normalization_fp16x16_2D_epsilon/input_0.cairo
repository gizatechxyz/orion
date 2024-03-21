use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 6567, sign: true });
    data.append(FP16x16 { mag: 14085, sign: true });
    data.append(FP16x16 { mag: 22357, sign: true });
    data.append(FP16x16 { mag: 118643, sign: true });
    data.append(FP16x16 { mag: 65923, sign: true });
    data.append(FP16x16 { mag: 144614, sign: true });
    data.append(FP16x16 { mag: 901, sign: true });
    data.append(FP16x16 { mag: 1674, sign: true });
    data.append(FP16x16 { mag: 66708, sign: false });
    data.append(FP16x16 { mag: 42929, sign: true });
    data.append(FP16x16 { mag: 23239, sign: true });
    data.append(FP16x16 { mag: 74402, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
