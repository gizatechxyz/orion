use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 57004, sign: false });
    data.append(FP16x16 { mag: 122231, sign: true });
    data.append(FP16x16 { mag: 24254, sign: true });
    data.append(FP16x16 { mag: 43862, sign: true });
    data.append(FP16x16 { mag: 109847, sign: true });
    data.append(FP16x16 { mag: 171231, sign: false });
    data.append(FP16x16 { mag: 143549, sign: true });
    data.append(FP16x16 { mag: 23240, sign: true });
    data.append(FP16x16 { mag: 200537, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
