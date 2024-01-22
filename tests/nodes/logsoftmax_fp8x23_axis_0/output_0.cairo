use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorSub};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 8617021, sign: true });
    data.append(FP8x23 { mag: 19680507, sign: true });
    data.append(FP8x23 { mag: 3717522, sign: true });
    data.append(FP8x23 { mag: 844233, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
