use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 196608, sign: false });
    data.append(FP16x16 { mag: 65536, sign: false });
    data.append(FP16x16 { mag: 262144, sign: false });
    data.append(FP16x16 { mag: 786432, sign: false });
    data.append(FP16x16 { mag: 589824, sign: false });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 1245184, sign: false });
    data.append(FP16x16 { mag: 1376256, sign: false });
    data.append(FP16x16 { mag: 1376256, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
