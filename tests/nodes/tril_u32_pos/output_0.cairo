use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{U32Tensor, U32TensorAdd};

fn output_0() -> Tensor<u32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(7);
    data.append(91);
    data.append(89);
    data.append(0);
    data.append(0);
    data.append(80);
    data.append(254);
    data.append(245);
    data.append(234);
    data.append(0);
    data.append(21);
    data.append(246);
    data.append(193);
    data.append(59);
    data.append(88);
    data.append(234);
    data.append(199);
    data.append(194);
    data.append(79);
    data.append(107);
    TensorTrait::new(shape.span(), data.span())
}
