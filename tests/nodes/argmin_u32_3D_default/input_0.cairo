use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(183);
    data.append(101);
    data.append(207);
    data.append(197);
    data.append(120);
    data.append(204);
    data.append(156);
    data.append(234);
    TensorTrait::new(shape.span(), data.span())
}
