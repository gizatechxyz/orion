use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 32, sign: false });
    data.append(i8 { mag: 39, sign: false });
    data.append(i8 { mag: 80, sign: true });
    data.append(i8 { mag: 108, sign: true });
    data.append(i8 { mag: 78, sign: false });
    data.append(i8 { mag: 69, sign: true });
    data.append(i8 { mag: 95, sign: false });
    data.append(i8 { mag: 5, sign: true });
    data.append(i8 { mag: 81, sign: false });
    data.append(i8 { mag: 88, sign: true });
    data.append(i8 { mag: 89, sign: true });
    data.append(i8 { mag: 10, sign: true });
    data.append(i8 { mag: 126, sign: false });
    data.append(i8 { mag: 53, sign: true });
    data.append(i8 { mag: 36, sign: true });
    data.append(i8 { mag: 14, sign: true });
    data.append(i8 { mag: 42, sign: false });
    data.append(i8 { mag: 20, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
