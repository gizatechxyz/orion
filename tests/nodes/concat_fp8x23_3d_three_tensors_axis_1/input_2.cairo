use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_2() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 452984832, sign: false });
    data.append(FP8x23 { mag: 461373440, sign: false });
    data.append(FP8x23 { mag: 469762048, sign: false });
    data.append(FP8x23 { mag: 478150656, sign: false });
    data.append(FP8x23 { mag: 486539264, sign: false });
    data.append(FP8x23 { mag: 494927872, sign: false });
    data.append(FP8x23 { mag: 503316480, sign: false });
    data.append(FP8x23 { mag: 511705088, sign: false });
    data.append(FP8x23 { mag: 520093696, sign: false });
    data.append(FP8x23 { mag: 528482304, sign: false });
    data.append(FP8x23 { mag: 536870912, sign: false });
    data.append(FP8x23 { mag: 545259520, sign: false });
    data.append(FP8x23 { mag: 553648128, sign: false });
    data.append(FP8x23 { mag: 562036736, sign: false });
    data.append(FP8x23 { mag: 570425344, sign: false });
    data.append(FP8x23 { mag: 578813952, sign: false });
    data.append(FP8x23 { mag: 587202560, sign: false });
    data.append(FP8x23 { mag: 595591168, sign: false });
    data.append(FP8x23 { mag: 603979776, sign: false });
    data.append(FP8x23 { mag: 612368384, sign: false });
    data.append(FP8x23 { mag: 620756992, sign: false });
    data.append(FP8x23 { mag: 629145600, sign: false });
    data.append(FP8x23 { mag: 637534208, sign: false });
    data.append(FP8x23 { mag: 645922816, sign: false });
    data.append(FP8x23 { mag: 654311424, sign: false });
    data.append(FP8x23 { mag: 662700032, sign: false });
    data.append(FP8x23 { mag: 671088640, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
