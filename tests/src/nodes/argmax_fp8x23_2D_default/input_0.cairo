use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 687865856, sign: false });
    data.append(FP8x23 { mag: 125829120, sign: false });
    data.append(FP8x23 { mag: 201326592, sign: true });
    data.append(FP8x23 { mag: 721420288, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
