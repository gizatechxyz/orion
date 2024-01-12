use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(191);
    data.append(83);
    data.append(144);
    data.append(69);
    data.append(77);
    data.append(34);
    TensorTrait::new(shape.span(), data.span())
}
