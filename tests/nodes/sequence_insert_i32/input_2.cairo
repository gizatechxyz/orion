use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_2() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 1, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
