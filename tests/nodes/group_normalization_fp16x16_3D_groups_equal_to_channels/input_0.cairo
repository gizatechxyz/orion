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
    data.append(FP16x16 { mag: 37822, sign: false });
    data.append(FP16x16 { mag: 3869, sign: true });
    data.append(FP16x16 { mag: 128605, sign: true });
    data.append(FP16x16 { mag: 49756, sign: true });
    data.append(FP16x16 { mag: 30741, sign: false });
    data.append(FP16x16 { mag: 1963, sign: false });
    data.append(FP16x16 { mag: 89126, sign: false });
    data.append(FP16x16 { mag: 4537, sign: false });
    data.append(FP16x16 { mag: 72047, sign: true });
    data.append(FP16x16 { mag: 97, sign: false });
    data.append(FP16x16 { mag: 50117, sign: false });
    data.append(FP16x16 { mag: 10199, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
