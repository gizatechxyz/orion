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
    data.append(FP16x16 { mag: 720896, sign: true });
    data.append(FP16x16 { mag: 4390912, sign: false });
    data.append(FP16x16 { mag: 5111808, sign: true });
    data.append(FP16x16 { mag: 6553600, sign: false });
    data.append(FP16x16 { mag: 1245184, sign: false });
    data.append(FP16x16 { mag: 6094848, sign: true });
    data.append(FP16x16 { mag: 4718592, sign: false });
    data.append(FP16x16 { mag: 3211264, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
