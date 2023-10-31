use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn input_2() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(54);
    data.append(55);
    data.append(56);
    data.append(57);
    data.append(58);
    data.append(59);
    data.append(60);
    data.append(61);
    data.append(62);
    data.append(63);
    data.append(64);
    data.append(65);
    data.append(66);
    data.append(67);
    data.append(68);
    data.append(69);
    data.append(70);
    data.append(71);
    data.append(72);
    data.append(73);
    data.append(74);
    data.append(75);
    data.append(76);
    data.append(77);
    data.append(78);
    data.append(79);
    data.append(80);
    TensorTrait::new(shape.span(), data.span())
}
