use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 333, sign: false });
    data.append(FP16x16 { mag: 31040, sign: true });
    data.append(FP16x16 { mag: 145280, sign: false });
    data.append(FP16x16 { mag: 64832, sign: true });
    data.append(FP16x16 { mag: 53856, sign: false });
    data.append(FP16x16 { mag: 183808, sign: false });
    data.append(FP16x16 { mag: 36128, sign: false });
    data.append(FP16x16 { mag: 22592, sign: false });
    data.append(FP16x16 { mag: 155392, sign: false });
    data.append(FP16x16 { mag: 79552, sign: false });
    data.append(FP16x16 { mag: 86144, sign: true });
    data.append(FP16x16 { mag: 42528, sign: false });
    data.append(FP16x16 { mag: 46112, sign: false });
    data.append(FP16x16 { mag: 54272, sign: true });
    data.append(FP16x16 { mag: 166272, sign: true });
    data.append(FP16x16 { mag: 39264, sign: false });
    data.append(FP16x16 { mag: 193792, sign: true });
    data.append(FP16x16 { mag: 2156, sign: true });
    data.append(FP16x16 { mag: 171904, sign: false });
    data.append(FP16x16 { mag: 125888, sign: true });
    data.append(FP16x16 { mag: 58880, sign: true });
    data.append(FP16x16 { mag: 134400, sign: true });
    data.append(FP16x16 { mag: 11376, sign: false });
    data.append(FP16x16 { mag: 32176, sign: false });
    data.append(FP16x16 { mag: 124864, sign: false });
    data.append(FP16x16 { mag: 47520, sign: true });
    data.append(FP16x16 { mag: 3046, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
