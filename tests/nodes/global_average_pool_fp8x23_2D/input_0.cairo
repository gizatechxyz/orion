use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 135376663, sign: true });
    data.append(FP8x23 { mag: 141510741, sign: true });
    data.append(FP8x23 { mag: 178081079, sign: true });
    data.append(FP8x23 { mag: 60683532, sign: true });
    data.append(FP8x23 { mag: 141357154, sign: false });
    data.append(FP8x23 { mag: 113733774, sign: false });
    data.append(FP8x23 { mag: 242689028, sign: true });
    data.append(FP8x23 { mag: 105029373, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
