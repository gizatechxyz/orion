use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(112);
    data.append(12);
    data.append(149);
    data.append(247);
    data.append(144);
    data.append(52);
    data.append(161);
    data.append(15);
    data.append(89);
    data.append(32);
    data.append(118);
    data.append(122);
    data.append(56);
    data.append(100);
    data.append(207);
    data.append(176);
    data.append(97);
    data.append(234);
    data.append(73);
    data.append(53);
    TensorTrait::new(shape.span(), data.span())
}
