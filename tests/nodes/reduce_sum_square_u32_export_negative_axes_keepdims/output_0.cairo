use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(462);
    data.append(525);
    data.append(594);
    data.append(669);
    data.append(750);
    data.append(837);
    data.append(930);
    data.append(1029);
    data.append(1134);
    TensorTrait::new(shape.span(), data.span())
}
