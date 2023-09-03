use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 1, sign: false });
    data.append(i8 { mag: 3, sign: false });
    data.append(i8 { mag: 6, sign: false });
    data.append(i8 { mag: 10, sign: false });
    data.append(i8 { mag: 15, sign: false });
    TensorTrait::new(shape.span(), data.span())
}