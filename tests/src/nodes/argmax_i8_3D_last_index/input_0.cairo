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
    data.append(i8 { mag: 116, sign: true });
    data.append(i8 { mag: 7, sign: true });
    data.append(i8 { mag: 10, sign: false });
    data.append(i8 { mag: 44, sign: false });
    data.append(i8 { mag: 122, sign: true });
    data.append(i8 { mag: 4, sign: true });
    data.append(i8 { mag: 87, sign: false });
    data.append(i8 { mag: 87, sign: false });
    TensorTrait::new(shape.span(), data.span())
}