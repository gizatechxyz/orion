use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 56236, sign: false });
    data.append(FP16x16 { mag: 61356, sign: false });
    data.append(FP16x16 { mag: 59267, sign: false });
    data.append(FP16x16 { mag: 7019, sign: false });
    data.append(FP16x16 { mag: 27265, sign: false });
    data.append(FP16x16 { mag: 35843, sign: false });
    data.append(FP16x16 { mag: 12233, sign: false });
    data.append(FP16x16 { mag: 47311, sign: false });
    data.append(FP16x16 { mag: 14312, sign: false });
    data.append(FP16x16 { mag: 3477, sign: false });
    data.append(FP16x16 { mag: 39621, sign: false });
    data.append(FP16x16 { mag: 44543, sign: false });
    data.append(FP16x16 { mag: 56785, sign: false });
    data.append(FP16x16 { mag: 29674, sign: false });
    data.append(FP16x16 { mag: 39650, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
