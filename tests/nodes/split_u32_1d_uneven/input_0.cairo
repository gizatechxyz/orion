use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(7);

    let mut data = ArrayTrait::new();
    data.append(203);
    data.append(159);
    data.append(108);
    data.append(166);
    data.append(98);
    data.append(220);
    data.append(233);
    TensorTrait::new(shape.span(), data.span())
}
