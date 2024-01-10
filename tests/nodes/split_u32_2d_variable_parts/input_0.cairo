use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(189);
    data.append(74);
    data.append(230);
    data.append(245);
    data.append(231);
    data.append(162);
    data.append(11);
    data.append(159);
    data.append(108);
    data.append(92);
    data.append(6);
    data.append(61);
    TensorTrait::new(shape.span(), data.span())
}
