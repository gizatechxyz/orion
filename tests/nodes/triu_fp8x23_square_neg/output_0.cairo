use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1040187392, sign: true });
    data.append(FP8x23 { mag: 360710144, sign: true });
    data.append(FP8x23 { mag: 327155712, sign: false });
    data.append(FP8x23 { mag: 461373440, sign: true });
    data.append(FP8x23 { mag: 352321536, sign: true });
    data.append(FP8x23 { mag: 570425344, sign: false });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 50331648, sign: false });
    data.append(FP8x23 { mag: 209715200, sign: false });
    data.append(FP8x23 { mag: 268435456, sign: false });
    data.append(FP8x23 { mag: 503316480, sign: true });
    data.append(FP8x23 { mag: 276824064, sign: true });
    data.append(FP8x23 { mag: 478150656, sign: true });
    data.append(FP8x23 { mag: 411041792, sign: false });
    data.append(FP8x23 { mag: 780140544, sign: true });
    data.append(FP8x23 { mag: 0, sign: false });
    data.append(FP8x23 { mag: 528482304, sign: true });
    data.append(FP8x23 { mag: 243269632, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
