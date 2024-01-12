use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 116327, sign: false });
    data.append(FP16x16 { mag: 204472, sign: false });
    data.append(FP16x16 { mag: 292618, sign: false });
    data.append(FP16x16 { mag: 468910, sign: false });
    data.append(FP16x16 { mag: 557056, sign: false });
    data.append(FP16x16 { mag: 645201, sign: false });
    data.append(FP16x16 { mag: 821493, sign: false });
    data.append(FP16x16 { mag: 909639, sign: false });
    data.append(FP16x16 { mag: 997785, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
