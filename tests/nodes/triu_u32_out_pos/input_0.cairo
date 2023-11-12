use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(193);
    data.append(144);
    data.append(205);
    data.append(145);
    data.append(118);
    data.append(23);
    data.append(17);
    data.append(87);
    data.append(243);
    data.append(169);
    data.append(151);
    data.append(57);
    data.append(211);
    data.append(152);
    data.append(175);
    data.append(47);
    data.append(201);
    data.append(223);
    data.append(42);
    data.append(101);
    TensorTrait::new(shape.span(), data.span())
}
