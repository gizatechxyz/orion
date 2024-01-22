use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorSub};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 867902390, sign: false });
    data.append(FP8x23 { mag: 179014618, sign: false });
    data.append(FP8x23 { mag: 1037532703, sign: false });
    data.append(FP8x23 { mag: 37869648, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
