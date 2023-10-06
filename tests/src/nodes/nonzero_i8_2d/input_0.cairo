use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 8, sign: true });
    data.append(i8 { mag: 44, sign: true });
    data.append(i8 { mag: 104, sign: true });
    data.append(i8 { mag: 21, sign: false });
    data.append(i8 { mag: 101, sign: false });
    data.append(i8 { mag: 84, sign: true });
    data.append(i8 { mag: 82, sign: false });
    data.append(i8 { mag: 34, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
