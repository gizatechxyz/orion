use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 243269632, sign: false });
    data.append(FP8x23 { mag: 520093696, sign: false });
    data.append(FP8x23 { mag: 58720256, sign: false });
    data.append(FP8x23 { mag: 494927872, sign: false });
    data.append(FP8x23 { mag: 780140544, sign: true });
    data.append(FP8x23 { mag: 452984832, sign: false });
    data.append(FP8x23 { mag: 746586112, sign: false });
    data.append(FP8x23 { mag: 276824064, sign: true });
    data.append(FP8x23 { mag: 838860800, sign: true });
    data.append(FP8x23 { mag: 478150656, sign: true });
    data.append(FP8x23 { mag: 989855744, sign: true });
    data.append(FP8x23 { mag: 788529152, sign: false });
    data.append(FP8x23 { mag: 763363328, sign: false });
    data.append(FP8x23 { mag: 528482304, sign: true });
    data.append(FP8x23 { mag: 50331648, sign: true });
    data.append(FP8x23 { mag: 545259520, sign: false });
    data.append(FP8x23 { mag: 310378496, sign: true });
    data.append(FP8x23 { mag: 704643072, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
