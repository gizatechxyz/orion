use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 16287022, sign: true });
    data.append(FP8x23 { mag: 10799553, sign: false });
    data.append(FP8x23 { mag: 2484402, sign: true });
    data.append(FP8x23 { mag: 17585288, sign: true });
    data.append(FP8x23 { mag: 6428, sign: false });
    data.append(FP8x23 { mag: 22747410, sign: false });
    data.append(FP8x23 { mag: 7024249, sign: false });
    data.append(FP8x23 { mag: 1080111, sign: false });
    data.append(FP8x23 { mag: 21293057, sign: true });
    data.append(FP8x23 { mag: 1501238, sign: true });
    data.append(FP8x23 { mag: 8554184, sign: true });
    data.append(FP8x23 { mag: 12577394, sign: false });
    data.append(FP8x23 { mag: 14241673, sign: true });
    data.append(FP8x23 { mag: 316469, sign: true });
    data.append(FP8x23 { mag: 16672164, sign: false });
    data.append(FP8x23 { mag: 23534429, sign: false });
    data.append(FP8x23 { mag: 22979924, sign: false });
    data.append(FP8x23 { mag: 12554544, sign: true });
    data.append(FP8x23 { mag: 8831121, sign: false });
    data.append(FP8x23 { mag: 12310986, sign: false });
    data.append(FP8x23 { mag: 16220051, sign: false });
    data.append(FP8x23 { mag: 1096465, sign: true });
    data.append(FP8x23 { mag: 1158077, sign: false });
    data.append(FP8x23 { mag: 7755965, sign: false });
    data.append(FP8x23 { mag: 24795265, sign: false });
    data.append(FP8x23 { mag: 1285412, sign: true });
    data.append(FP8x23 { mag: 12210140, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
