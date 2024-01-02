use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorSub};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 12134368, sign: false });
    data.append(FP8x23 { mag: 22252681, sign: true });
    data.append(FP8x23 { mag: 1818273, sign: false });
    data.append(FP8x23 { mag: 18287369, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
