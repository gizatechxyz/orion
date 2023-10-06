use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 2097152, sign: false });
    data.append(FP16x16 { mag: 5111808, sign: true });
    data.append(FP16x16 { mag: 8126464, sign: true });
    data.append(FP16x16 { mag: 4194304, sign: true });
    data.append(FP16x16 { mag: 2424832, sign: false });
    data.append(FP16x16 { mag: 7340032, sign: true });
    data.append(FP16x16 { mag: 5242880, sign: false });
    data.append(FP16x16 { mag: 1179648, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
