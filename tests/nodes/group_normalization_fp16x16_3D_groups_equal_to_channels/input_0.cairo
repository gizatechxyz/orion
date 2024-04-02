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
    data.append(FP16x16 { mag: 121219, sign: true });
    data.append(FP16x16 { mag: 163478, sign: true });
    data.append(FP16x16 { mag: 12025, sign: false });
    data.append(FP16x16 { mag: 30821, sign: true });
    data.append(FP16x16 { mag: 24253, sign: true });
    data.append(FP16x16 { mag: 11014, sign: true });
    data.append(FP16x16 { mag: 11181, sign: false });
    data.append(FP16x16 { mag: 48248, sign: true });
    data.append(FP16x16 { mag: 42824, sign: false });
    data.append(FP16x16 { mag: 83487, sign: false });
    data.append(FP16x16 { mag: 13949, sign: true });
    data.append(FP16x16 { mag: 18405, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
