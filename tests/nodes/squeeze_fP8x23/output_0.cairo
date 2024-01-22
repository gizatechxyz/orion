use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorSub};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1149239296, sign: false });
    data.append(FP8x23 { mag: 1912602624, sign: false });
    data.append(FP8x23 { mag: 1887436800, sign: false });
    data.append(FP8x23 { mag: 813694976, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
