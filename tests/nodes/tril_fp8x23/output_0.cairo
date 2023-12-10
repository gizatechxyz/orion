use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 234881024, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 92274688, sign: true });
    data.append(FP8x23 { mag: 998244352, sign: true });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 494927872, sign: false });
    data.append(FP8x23 { mag: 394264576, sign: false });
    data.append(FP8x23 { mag: 83886080, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 654311424, sign: true });
    data.append(FP8x23 { mag: 754974720, sign: true });
    data.append(FP8x23 { mag: 1023410176, sign: false });
    data.append(FP8x23 { mag: 520093696, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
