use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 6684672, sign: false });
    data.append(FP16x16 { mag: 6094848, sign: false });
    data.append(FP16x16 { mag: 7471104, sign: true });
    data.append(FP16x16 { mag: 2031616, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
