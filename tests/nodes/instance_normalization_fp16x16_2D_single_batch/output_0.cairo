use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 29824, sign: true });
    data.append(FP16x16 { mag: 48933, sign: true });
    data.append(FP16x16 { mag: 23720, sign: true });
    data.append(FP16x16 { mag: 145178, sign: false });
    data.append(FP16x16 { mag: 36572, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
