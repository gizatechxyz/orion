use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 528482304, sign: false });
    data.append(FP8x23 { mag: 780140544, sign: true });
    data.append(FP8x23 { mag: 830472192, sign: false });
    data.append(FP8x23 { mag: 75497472, sign: true });
    data.append(FP8x23 { mag: 100663296, sign: false });
    data.append(FP8x23 { mag: 452984832, sign: false });
    data.append(FP8x23 { mag: 897581056, sign: true });
    data.append(FP8x23 { mag: 385875968, sign: false });
    data.append(FP8x23 { mag: 150994944, sign: true });
    data.append(FP8x23 { mag: 385875968, sign: false });
    data.append(FP8x23 { mag: 411041792, sign: false });
    data.append(FP8x23 { mag: 301989888, sign: true });
    data.append(FP8x23 { mag: 444596224, sign: true });
    data.append(FP8x23 { mag: 905969664, sign: false });
    data.append(FP8x23 { mag: 251658240, sign: false });
    data.append(FP8x23 { mag: 830472192, sign: false });
    data.append(FP8x23 { mag: 864026624, sign: true });
    data.append(FP8x23 { mag: 545259520, sign: true });
    data.append(FP8x23 { mag: 1040187392, sign: false });
    data.append(FP8x23 { mag: 243269632, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
