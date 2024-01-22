use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 981467136, sign: true });
    data.append(FP8x23 { mag: 914358272, sign: false });
    data.append(FP8x23 { mag: 662700032, sign: true });
    data.append(FP8x23 { mag: 58720256, sign: false });
    data.append(FP8x23 { mag: 176160768, sign: false });
    data.append(FP8x23 { mag: 905969664, sign: true });
    data.append(FP8x23 { mag: 486539264, sign: true });
    data.append(FP8x23 { mag: 838860800, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
