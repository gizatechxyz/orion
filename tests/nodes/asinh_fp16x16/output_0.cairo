use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 110173, sign: false });
    data.append(FP16x16 { mag: 135989, sign: false });
    data.append(FP16x16 { mag: 136114, sign: false });
    data.append(FP16x16 { mag: 116149, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
