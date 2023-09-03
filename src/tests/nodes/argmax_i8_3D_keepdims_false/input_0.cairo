use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 25, sign: true });
    data.append(i8 { mag: 88, sign: false });
    data.append(i8 { mag: 44, sign: true });
    data.append(i8 { mag: 124, sign: false });
    data.append(i8 { mag: 35, sign: false });
    data.append(i8 { mag: 21, sign: false });
    data.append(i8 { mag: 4, sign: true });
    data.append(i8 { mag: 39, sign: false });
    TensorTrait::new(shape.span(), data.span())
}