use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(3);
    data.append(4);
    data.append(5);
    data.append(6);
    data.append(7);
    data.append(8);
    data.append(15);
    data.append(16);
    data.append(17);
    data.append(18);
    data.append(19);
    data.append(20);
    data.append(27);
    data.append(28);
    data.append(29);
    data.append(30);
    data.append(31);
    data.append(32);
    TensorTrait::new(shape.span(), data.span())
}
