use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 62, sign: true });
    data.append(i8 { mag: 15, sign: true });
    data.append(i8 { mag: 60, sign: true });
    data.append(i8 { mag: 122, sign: true });
    data.append(i8 { mag: 5, sign: true });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 59, sign: false });
    data.append(i8 { mag: 113, sign: false });
    data.append(i8 { mag: 126, sign: true });
    data.append(i8 { mag: 1, sign: true });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 93, sign: true });
    data.append(i8 { mag: 60, sign: false });
    data.append(i8 { mag: 118, sign: false });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 25, sign: true });
    data.append(i8 { mag: 111, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
