use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 637534208, sign: true });
    data.append(FP8x23 { mag: 293601280, sign: true });
    data.append(FP8x23 { mag: 620756992, sign: true });
    data.append(FP8x23 { mag: 25165824, sign: true });
    data.append(FP8x23 { mag: 150994944, sign: false });
    data.append(FP8x23 { mag: 452984832, sign: false });
    data.append(FP8x23 { mag: 780140544, sign: false });
    data.append(FP8x23 { mag: 528482304, sign: true });
    data.append(FP8x23 { mag: 1023410176, sign: true });
    data.append(FP8x23 { mag: 553648128, sign: false });
    data.append(FP8x23 { mag: 587202560, sign: false });
    data.append(FP8x23 { mag: 998244352, sign: true });
    data.append(FP8x23 { mag: 671088640, sign: true });
    data.append(FP8x23 { mag: 796917760, sign: true });
    data.append(FP8x23 { mag: 578813952, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
