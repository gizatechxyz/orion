use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 293601280, sign: false });
    data.append(FP8x23 { mag: 402653184, sign: true });
    data.append(FP8x23 { mag: 629145600, sign: true });
    data.append(FP8x23 { mag: 838860800, sign: false });
    data.append(FP8x23 { mag: 1065353216, sign: true });
    data.append(FP8x23 { mag: 553648128, sign: true });
    data.append(FP8x23 { mag: 914358272, sign: true });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 796917760, sign: false });
    data.append(FP8x23 { mag: 981467136, sign: false });
    data.append(FP8x23 { mag: 83886080, sign: false });
    data.append(FP8x23 { mag: 847249408, sign: false });
    data.append(FP8x23 { mag: 704643072, sign: false });
    data.append(FP8x23 { mag: 746586112, sign: true });
    data.append(FP8x23 { mag: 998244352, sign: false });
    data.append(FP8x23 { mag: 125829120, sign: true });
    data.append(FP8x23 { mag: 914358272, sign: false });
    data.append(FP8x23 { mag: 603979776, sign: true });
    data.append(FP8x23 { mag: 872415232, sign: false });
    data.append(FP8x23 { mag: 872415232, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
