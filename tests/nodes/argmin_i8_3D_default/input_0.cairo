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
    data.append(i8 { mag: 109, sign: false });
    data.append(i8 { mag: 11, sign: false });
    data.append(i8 { mag: 107, sign: true });
    data.append(i8 { mag: 97, sign: true });
    data.append(i8 { mag: 110, sign: true });
    data.append(i8 { mag: 118, sign: false });
    data.append(i8 { mag: 114, sign: true });
    data.append(i8 { mag: 70, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
