use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorSub};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 192937984, sign: true });
    data.append(FP8x23 { mag: 729808896, sign: false });
    data.append(FP8x23 { mag: 637534208, sign: false });
    data.append(FP8x23 { mag: 830472192, sign: true });
    data.append(FP8x23 { mag: 931135488, sign: true });
    data.append(FP8x23 { mag: 58720256, sign: false });
    data.append(FP8x23 { mag: 595591168, sign: true });
    data.append(FP8x23 { mag: 520093696, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
