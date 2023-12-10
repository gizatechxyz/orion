use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 1, sign: true });
    data.append(i32 { mag: 114, sign: true });
    data.append(i32 { mag: 68, sign: false });
    data.append(i32 { mag: 49, sign: false });
    data.append(i32 { mag: 105, sign: false });
    data.append(i32 { mag: 115, sign: true });
    data.append(i32 { mag: 54, sign: true });
    data.append(i32 { mag: 110, sign: true });
    data.append(i32 { mag: 83, sign: true });
    data.append(i32 { mag: 41, sign: true });
    data.append(i32 { mag: 66, sign: false });
    data.append(i32 { mag: 34, sign: false });
    data.append(i32 { mag: 67, sign: false });
    data.append(i32 { mag: 57, sign: false });
    data.append(i32 { mag: 58, sign: false });
    data.append(i32 { mag: 3, sign: false });
    data.append(i32 { mag: 115, sign: true });
    data.append(i32 { mag: 2, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
