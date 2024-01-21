use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 66655, sign: false });
    data.append(FP16x16 { mag: 234574, sign: true });
    data.append(FP16x16 { mag: 36595, sign: false });
    data.append(FP16x16 { mag: 43544, sign: false });
    data.append(FP16x16 { mag: 56268, sign: true });
    data.append(FP16x16 { mag: 203346, sign: true });
    data.append(FP16x16 { mag: 4468, sign: false });
    data.append(FP16x16 { mag: 60114, sign: true });
    data.append(FP16x16 { mag: 19585, sign: true });
    data.append(FP16x16 { mag: 118482, sign: true });
    data.append(FP16x16 { mag: 24747, sign: true });
    data.append(FP16x16 { mag: 130887, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
