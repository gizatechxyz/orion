use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 99, sign: false });
    data.append(i32 { mag: 97, sign: false });
    data.append(i32 { mag: 55, sign: false });
    data.append(i32 { mag: 13, sign: false });
    data.append(i32 { mag: 31, sign: true });
    data.append(i32 { mag: 120, sign: false });
    data.append(i32 { mag: 34, sign: false });
    data.append(i32 { mag: 70, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
