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
    data.append(FP16x16 { mag: 56389, sign: true });
    data.append(FP16x16 { mag: 81104, sign: true });
    data.append(FP16x16 { mag: 105842, sign: true });
    data.append(FP16x16 { mag: 29201, sign: true });
    data.append(FP16x16 { mag: 80650, sign: true });
    data.append(FP16x16 { mag: 33732, sign: false });
    data.append(FP16x16 { mag: 44311, sign: true });
    data.append(FP16x16 { mag: 44217, sign: true });
    data.append(FP16x16 { mag: 38155, sign: false });
    data.append(FP16x16 { mag: 11275, sign: true });
    data.append(FP16x16 { mag: 53819, sign: true });
    data.append(FP16x16 { mag: 20372, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
