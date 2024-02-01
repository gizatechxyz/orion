use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(10);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 0, sign: false });
    data.append(FP16x16 { mag: 7656, sign: false });
    data.append(FP16x16 { mag: 27056, sign: false });
    data.append(FP16x16 { mag: 49120, sign: false });
    data.append(FP16x16 { mag: 63552, sign: false });
    data.append(FP16x16 { mag: 63552, sign: false });
    data.append(FP16x16 { mag: 49184, sign: false });
    data.append(FP16x16 { mag: 27104, sign: false });
    data.append(FP16x16 { mag: 7732, sign: false });
    data.append(FP16x16 { mag: 0, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
