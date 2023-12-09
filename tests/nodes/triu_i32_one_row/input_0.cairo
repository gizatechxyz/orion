use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 7, sign: false });
    data.append(i32 { mag: 32, sign: false });
    data.append(i32 { mag: 83, sign: false });
    data.append(i32 { mag: 52, sign: false });
    data.append(i32 { mag: 49, sign: true });
    data.append(i32 { mag: 23, sign: false });
    data.append(i32 { mag: 92, sign: true });
    data.append(i32 { mag: 97, sign: false });
    data.append(i32 { mag: 24, sign: true });
    data.append(i32 { mag: 46, sign: false });
    data.append(i32 { mag: 67, sign: true });
    data.append(i32 { mag: 108, sign: true });
    data.append(i32 { mag: 10, sign: true });
    data.append(i32 { mag: 119, sign: false });
    data.append(i32 { mag: 109, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
