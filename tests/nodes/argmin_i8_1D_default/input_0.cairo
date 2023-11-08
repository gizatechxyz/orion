use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 62, sign: false });
    data.append(i8 { mag: 69, sign: false });
    data.append(i8 { mag: 94, sign: false });
    TensorTrait::new(shape.span(), data.span())
}