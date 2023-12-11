use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 171911, sign: false });
    data.append(FP16x16 { mag: 125876, sign: true });
    data.append(FP16x16 { mag: 58883, sign: true });
    data.append(FP16x16 { mag: 134443, sign: true });
    data.append(FP16x16 { mag: 11378, sign: false });
    data.append(FP16x16 { mag: 32169, sign: false });
    data.append(FP16x16 { mag: 124876, sign: false });
    data.append(FP16x16 { mag: 47530, sign: true });
    data.append(FP16x16 { mag: 3046, sign: false });
    data.append(FP16x16 { mag: 79544, sign: false });
    data.append(FP16x16 { mag: 86148, sign: true });
    data.append(FP16x16 { mag: 42525, sign: false });
    data.append(FP16x16 { mag: 46115, sign: false });
    data.append(FP16x16 { mag: 54280, sign: true });
    data.append(FP16x16 { mag: 166328, sign: true });
    data.append(FP16x16 { mag: 39251, sign: false });
    data.append(FP16x16 { mag: 193736, sign: true });
    data.append(FP16x16 { mag: 2156, sign: true });
    data.append(FP16x16 { mag: 333, sign: false });
    data.append(FP16x16 { mag: 31045, sign: true });
    data.append(FP16x16 { mag: 145308, sign: false });
    data.append(FP16x16 { mag: 64842, sign: true });
    data.append(FP16x16 { mag: 53870, sign: false });
    data.append(FP16x16 { mag: 183784, sign: false });
    data.append(FP16x16 { mag: 36113, sign: false });
    data.append(FP16x16 { mag: 22596, sign: false });
    data.append(FP16x16 { mag: 155378, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
