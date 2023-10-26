use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn input_1() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 3, sign: false });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 2, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
