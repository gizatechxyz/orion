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
    data.append(i8 { mag: 123, sign: false });
    data.append(i8 { mag: 17, sign: false });
    data.append(i8 { mag: 34, sign: true });
    data.append(i8 { mag: 12, sign: true });
    data.append(i8 { mag: 112, sign: false });
    data.append(i8 { mag: 10, sign: false });
    data.append(i8 { mag: 66, sign: false });
    data.append(i8 { mag: 49, sign: false });
    TensorTrait::new(shape.span(), data.span())
}