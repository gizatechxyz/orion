use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(139);
    data.append(156);
    data.append(247);
    data.append(79);
    data.append(199);
    data.append(238);
    data.append(54);
    data.append(139);
    TensorTrait::new(shape.span(), data.span())
}
