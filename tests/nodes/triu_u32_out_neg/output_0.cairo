use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::U32Tensor;

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(220);
    data.append(57);
    data.append(12);
    data.append(49);
    data.append(133);
    data.append(203);
    data.append(211);
    data.append(116);
    data.append(97);
    data.append(161);
    data.append(136);
    data.append(68);
    data.append(121);
    data.append(175);
    data.append(87);
    data.append(36);
    data.append(86);
    data.append(11);
    data.append(85);
    data.append(82);
    TensorTrait::new(shape.span(), data.span())
}
