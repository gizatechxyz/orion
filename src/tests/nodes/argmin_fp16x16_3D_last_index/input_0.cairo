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
    data.append(FP16x16 { mag: 5898240, sign: true });
    data.append(FP16x16 { mag: 7471104, sign: false });
    data.append(FP16x16 { mag: 4063232, sign: true });
    data.append(FP16x16 { mag: 5111808, sign: true });
    data.append(FP16x16 { mag: 1310720, sign: true });
    data.append(FP16x16 { mag: 262144, sign: true });
    data.append(FP16x16 { mag: 5570560, sign: true });
    data.append(FP16x16 { mag: 7274496, sign: true });
    TensorTrait::new(shape.span(), data.span())
}