use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 6422528, sign: true });
    data.append(FP16x16 { mag: 1245184, sign: false });
    data.append(FP16x16 { mag: 6356992, sign: false });
    data.append(FP16x16 { mag: 4718592, sign: true });
    data.append(FP16x16 { mag: 6488064, sign: true });
    data.append(FP16x16 { mag: 4390912, sign: true });
    data.append(FP16x16 { mag: 5046272, sign: false });
    data.append(FP16x16 { mag: 2752512, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
