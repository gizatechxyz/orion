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
    data.append(FP8x23 { mag: 20907772, sign: true });
    data.append(FP8x23 { mag: 24683986, sign: true });
    data.append(FP8x23 { mag: 21494604, sign: true });
    data.append(FP8x23 { mag: 1609367, sign: false });
    TensorTrait::new(shape.span(), data.span())
}