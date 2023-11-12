use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 1, sign: false });
    data.append(FP16x16 { mag: 2, sign: false });
    data.append(FP16x16 { mag: 3, sign: false });
    data.append(FP16x16 { mag: 4, sign: false });
    data.append(FP16x16 { mag: 5, sign: false });
    data.append(FP16x16 { mag: 6, sign: false });
    data.append(FP16x16 { mag: 7, sign: false });
    data.append(FP16x16 { mag: 8, sign: false });
    data.append(FP16x16 { mag: 9, sign: false });
    data.append(FP16x16 { mag: 10, sign: false });
    data.append(FP16x16 { mag: 11, sign: false });
    data.append(FP16x16 { mag: 12, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
