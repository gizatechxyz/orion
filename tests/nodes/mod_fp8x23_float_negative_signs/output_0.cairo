use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 15758929, sign: true });
    data.append(FP8x23 { mag: 4062516, sign: true });
    data.append(FP8x23 { mag: 5339589, sign: true });
    data.append(FP8x23 { mag: 21255586, sign: true });
    data.append(FP8x23 { mag: 47554872, sign: true });
    data.append(FP8x23 { mag: 75464856, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
