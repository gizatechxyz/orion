use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 520093696, sign: false });
    data.append(FP8x23 { mag: 444596224, sign: false });
    data.append(FP8x23 { mag: 637534208, sign: false });
    data.append(FP8x23 { mag: 729808896, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
