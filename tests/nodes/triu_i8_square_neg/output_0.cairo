use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 127, sign: true });
    data.append(i8 { mag: 48, sign: true });
    data.append(i8 { mag: 39, sign: true });
    data.append(i8 { mag: 6, sign: false });
    data.append(i8 { mag: 15, sign: true });
    data.append(i8 { mag: 64, sign: true });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 70, sign: true });
    data.append(i8 { mag: 25, sign: true });
    data.append(i8 { mag: 52, sign: false });
    data.append(i8 { mag: 97, sign: true });
    data.append(i8 { mag: 53, sign: true });
    data.append(i8 { mag: 36, sign: false });
    data.append(i8 { mag: 117, sign: false });
    data.append(i8 { mag: 93, sign: true });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 56, sign: false });
    data.append(i8 { mag: 78, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
