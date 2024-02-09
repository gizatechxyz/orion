use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(1);
    shape.append(4);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 1376256, sign: false });
    data.append(FP16x16 { mag: 2162688, sign: false });
    data.append(FP16x16 { mag: 6488064, sign: false });
    data.append(FP16x16 { mag: 7667712, sign: false });
    data.append(FP16x16 { mag: 12386304, sign: false });
    data.append(FP16x16 { mag: 13565952, sign: false });
    data.append(FP16x16 { mag: 11206656, sign: false });
    data.append(FP16x16 { mag: 11993088, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
