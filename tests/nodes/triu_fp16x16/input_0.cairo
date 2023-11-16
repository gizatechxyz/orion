use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 6619136, sign: false });
    data.append(FP16x16 { mag: 4784128, sign: true });
    data.append(FP16x16 { mag: 1703936, sign: false });
    data.append(FP16x16 { mag: 5963776, sign: false });
    data.append(FP16x16 { mag: 2293760, sign: true });
    data.append(FP16x16 { mag: 917504, sign: false });
    data.append(FP16x16 { mag: 2424832, sign: true });
    data.append(FP16x16 { mag: 2883584, sign: true });
    data.append(FP16x16 { mag: 1966080, sign: false });
    data.append(FP16x16 { mag: 7208960, sign: true });
    data.append(FP16x16 { mag: 196608, sign: false });
    data.append(FP16x16 { mag: 7471104, sign: true });
    data.append(FP16x16 { mag: 6684672, sign: true });
    data.append(FP16x16 { mag: 7208960, sign: true });
    data.append(FP16x16 { mag: 7733248, sign: true });
    data.append(FP16x16 { mag: 1900544, sign: false });
    data.append(FP16x16 { mag: 6684672, sign: false });
    data.append(FP16x16 { mag: 3866624, sign: true });
    data.append(FP16x16 { mag: 7733248, sign: false });
    data.append(FP16x16 { mag: 4849664, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
