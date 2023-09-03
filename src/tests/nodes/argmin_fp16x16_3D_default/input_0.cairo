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
    data.append(FP16x16 { mag: 7798784, sign: false });
    data.append(FP16x16 { mag: 6422528, sign: false });
    data.append(FP16x16 { mag: 4587520, sign: true });
    data.append(FP16x16 { mag: 4390912, sign: true });
    data.append(FP16x16 { mag: 458752, sign: true });
    data.append(FP16x16 { mag: 4784128, sign: true });
    data.append(FP16x16 { mag: 3735552, sign: true });
    data.append(FP16x16 { mag: 3538944, sign: false });
    TensorTrait::new(shape.span(), data.span())
}