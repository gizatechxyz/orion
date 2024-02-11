use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(7);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 983040, sign: false });
    data.append(FP16x16 { mag: 6815744, sign: true });
    data.append(FP16x16 { mag: 458752, sign: true });
    data.append(FP16x16 { mag: 5439488, sign: true });
    data.append(FP16x16 { mag: 1441792, sign: false });
    data.append(FP16x16 { mag: 3211264, sign: false });
    data.append(FP16x16 { mag: 5373952, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
