use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_2() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 42958, sign: false });
    data.append(FP16x16 { mag: 57252, sign: false });
    data.append(FP16x16 { mag: 44948, sign: false });
    data.append(FP16x16 { mag: 18261, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
