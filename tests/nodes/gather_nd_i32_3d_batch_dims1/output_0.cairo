use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::I32Tensor;
use orion::numbers::{IntegerTrait, i32};

fn output_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 8, sign: false });
    data.append(i32 { mag: 9, sign: false });
    data.append(i32 { mag: 48, sign: false });
    data.append(i32 { mag: 36, sign: false });
    data.append(i32 { mag: 62, sign: false });
    data.append(i32 { mag: 69, sign: false });
    data.append(i32 { mag: 90, sign: false });
    data.append(i32 { mag: 87, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
