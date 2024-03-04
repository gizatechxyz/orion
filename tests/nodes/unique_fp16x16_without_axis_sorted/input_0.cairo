use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 66908, sign: false });
    data.append(FP16x16 { mag: 102211, sign: false });
    data.append(FP16x16 { mag: 31626, sign: false });
    data.append(FP16x16 { mag: 71039, sign: false });
    data.append(FP16x16 { mag: 18914, sign: false });
    data.append(FP16x16 { mag: 65772, sign: false });
    data.append(FP16x16 { mag: 88563, sign: false });
    data.append(FP16x16 { mag: 150507, sign: false });
    data.append(FP16x16 { mag: 152608, sign: false });
    data.append(FP16x16 { mag: 34995, sign: false });
    data.append(FP16x16 { mag: 113399, sign: false });
    data.append(FP16x16 { mag: 80984, sign: false });
    data.append(FP16x16 { mag: 67061, sign: false });
    data.append(FP16x16 { mag: 79121, sign: false });
    data.append(FP16x16 { mag: 196064, sign: false });
    data.append(FP16x16 { mag: 192947, sign: false });
    data.append(FP16x16 { mag: 16692, sign: false });
    data.append(FP16x16 { mag: 106660, sign: false });
    data.append(FP16x16 { mag: 94130, sign: false });
    data.append(FP16x16 { mag: 50303, sign: false });
    data.append(FP16x16 { mag: 28219, sign: false });
    data.append(FP16x16 { mag: 16416, sign: false });
    data.append(FP16x16 { mag: 193457, sign: false });
    data.append(FP16x16 { mag: 141602, sign: false });
    data.append(FP16x16 { mag: 73709, sign: false });
    data.append(FP16x16 { mag: 185408, sign: false });
    data.append(FP16x16 { mag: 194771, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
