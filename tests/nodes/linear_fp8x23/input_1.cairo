use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_1() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 52315818, sign: false });
    data.append(FP8x23 { mag: 2320148, sign: false });
    data.append(FP8x23 { mag: 22884193, sign: true });
    data.append(FP8x23 { mag: 21749272, sign: false });
    data.append(FP8x23 { mag: 29980322, sign: true });
    data.append(FP8x23 { mag: 34817611, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
