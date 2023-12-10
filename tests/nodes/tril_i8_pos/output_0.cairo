use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 45, sign: false });
    data.append(i8 { mag: 2, sign: true });
    data.append(i8 { mag: 2, sign: true });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 25, sign: false });
    data.append(i8 { mag: 33, sign: false });
    data.append(i8 { mag: 104, sign: true });
    data.append(i8 { mag: 70, sign: true });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 0, sign: false });
    data.append(i8 { mag: 84, sign: false });
    data.append(i8 { mag: 73, sign: true });
    data.append(i8 { mag: 22, sign: false });
    data.append(i8 { mag: 113, sign: true });
    data.append(i8 { mag: 21, sign: true });
    data.append(i8 { mag: 32, sign: true });
    data.append(i8 { mag: 6, sign: false });
    data.append(i8 { mag: 73, sign: false });
    data.append(i8 { mag: 15, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
