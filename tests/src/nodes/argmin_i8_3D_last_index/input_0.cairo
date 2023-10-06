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
    data.append(i8 { mag: 48, sign: false });
    data.append(i8 { mag: 34, sign: false });
    data.append(i8 { mag: 46, sign: true });
    data.append(i8 { mag: 113, sign: true });
    data.append(i8 { mag: 94, sign: true });
    data.append(i8 { mag: 30, sign: true });
    data.append(i8 { mag: 98, sign: false });
    data.append(i8 { mag: 114, sign: true });
    TensorTrait::new(shape.span(), data.span())
}