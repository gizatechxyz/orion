use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I8Tensor;
use orion::numbers::{IntegerTrait, i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 5, sign: true });
    data.append(i8 { mag: 84, sign: true });
    data.append(i8 { mag: 21, sign: false });
    data.append(i8 { mag: 107, sign: false });
    data.append(i8 { mag: 78, sign: true });
    data.append(i8 { mag: 100, sign: true });
    data.append(i8 { mag: 90, sign: true });
    data.append(i8 { mag: 69, sign: false });
    data.append(i8 { mag: 122, sign: false });
    data.append(i8 { mag: 97, sign: false });
    data.append(i8 { mag: 92, sign: true });
    data.append(i8 { mag: 63, sign: false });
    data.append(i8 { mag: 27, sign: false });
    data.append(i8 { mag: 83, sign: true });
    data.append(i8 { mag: 21, sign: false });
    data.append(i8 { mag: 75, sign: true });
    data.append(i8 { mag: 51, sign: true });
    data.append(i8 { mag: 61, sign: false });
    data.append(i8 { mag: 3, sign: true });
    data.append(i8 { mag: 25, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
