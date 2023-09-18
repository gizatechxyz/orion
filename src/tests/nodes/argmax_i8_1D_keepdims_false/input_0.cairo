use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 78, sign: false });
    data.append(i8 { mag: 118, sign: true });
    data.append(i8 { mag: 52, sign: true });
    TensorTrait::new(shape.span(), data.span())
}