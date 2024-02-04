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
    data.append(FP16x16 { mag: 196450, sign: false });
    data.append(FP16x16 { mag: 110458, sign: true });
    data.append(FP16x16 { mag: 25601, sign: true });
    data.append(FP16x16 { mag: 88269, sign: false });
    data.append(FP16x16 { mag: 9547, sign: false });
    data.append(FP16x16 { mag: 15522, sign: true });
    data.append(FP16x16 { mag: 59558, sign: true });
    data.append(FP16x16 { mag: 141551, sign: true });
    data.append(FP16x16 { mag: 136437, sign: false });
    data.append(FP16x16 { mag: 30273, sign: false });
    data.append(FP16x16 { mag: 157482, sign: false });
    data.append(FP16x16 { mag: 180474, sign: false });
    data.append(FP16x16 { mag: 16595, sign: false });
    data.append(FP16x16 { mag: 52197, sign: true });
    data.append(FP16x16 { mag: 90707, sign: false });
    data.append(FP16x16 { mag: 61283, sign: true });
    data.append(FP16x16 { mag: 12204, sign: true });
    data.append(FP16x16 { mag: 9568, sign: true });
    data.append(FP16x16 { mag: 98355, sign: false });
    data.append(FP16x16 { mag: 125547, sign: false });
    data.append(FP16x16 { mag: 108859, sign: true });
    data.append(FP16x16 { mag: 120511, sign: true });
    data.append(FP16x16 { mag: 109130, sign: true });
    data.append(FP16x16 { mag: 141659, sign: true });
    data.append(FP16x16 { mag: 25063, sign: false });
    data.append(FP16x16 { mag: 71249, sign: false });
    data.append(FP16x16 { mag: 47160, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
