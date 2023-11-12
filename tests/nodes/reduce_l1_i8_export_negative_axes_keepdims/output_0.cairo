use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 30, sign: false });
    data.append(i8 { mag: 33, sign: false });
    data.append(i8 { mag: 36, sign: false });
    data.append(i8 { mag: 39, sign: false });
    data.append(i8 { mag: 42, sign: false });
    data.append(i8 { mag: 45, sign: false });
    data.append(i8 { mag: 48, sign: false });
    data.append(i8 { mag: 51, sign: false });
    data.append(i8 { mag: 54, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
