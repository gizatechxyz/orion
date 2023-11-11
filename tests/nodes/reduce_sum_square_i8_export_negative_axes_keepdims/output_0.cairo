use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 26, sign: false });
    data.append(i8 { mag: 40, sign: false });
    data.append(i8 { mag: 58, sign: false });
    data.append(i8 { mag: 80, sign: false });
    TensorTrait::new(shape.span(), data.span())
}