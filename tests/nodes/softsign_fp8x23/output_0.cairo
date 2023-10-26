use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 7123934, sign: false });
    data.append(FP8x23 { mag: 7202457, sign: false });
    data.append(FP8x23 { mag: 3399107, sign: true });
    data.append(FP8x23 { mag: 5457715, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
