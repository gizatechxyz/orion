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
    data.append(i32 { mag: 69, sign: true });
    data.append(i32 { mag: 117, sign: false });
    data.append(i32 { mag: 92, sign: true });
    data.append(i32 { mag: 7, sign: false });
    data.append(i32 { mag: 33, sign: false });
    data.append(i32 { mag: 99, sign: true });
    data.append(i32 { mag: 86, sign: false });
    data.append(i32 { mag: 33, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
