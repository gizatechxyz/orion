use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 103, sign: false });
    data.append(i32 { mag: 25, sign: false });
    data.append(i32 { mag: 34, sign: true });
    data.append(i32 { mag: 43, sign: true });
    data.append(i32 { mag: 49, sign: false });
    data.append(i32 { mag: 106, sign: true });
    data.append(i32 { mag: 104, sign: true });
    data.append(i32 { mag: 105, sign: true });
    data.append(i32 { mag: 97, sign: true });
    data.append(i32 { mag: 126, sign: false });
    data.append(i32 { mag: 32, sign: false });
    data.append(i32 { mag: 120, sign: false });
    data.append(i32 { mag: 70, sign: true });
    data.append(i32 { mag: 77, sign: true });
    data.append(i32 { mag: 14, sign: true });
    data.append(i32 { mag: 11, sign: true });
    data.append(i32 { mag: 61, sign: false });
    data.append(i32 { mag: 117, sign: false });
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 58, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
