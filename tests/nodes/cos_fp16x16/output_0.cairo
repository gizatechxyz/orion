use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 39263, sign: true });
    data.append(FP16x16 { mag: 32755, sign: false });
    data.append(FP16x16 { mag: 64011, sign: true });
    data.append(FP16x16 { mag: 65152, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
