use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorSub};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 285212672, sign: true });
    data.append(FP8x23 { mag: 142606336, sign: true });
    data.append(FP8x23 { mag: 360710144, sign: false });
    data.append(FP8x23 { mag: 469762048, sign: false });
    data.append(FP8x23 { mag: 452984832, sign: false });
    data.append(FP8x23 { mag: 763363328, sign: true });
    data.append(FP8x23 { mag: 721420288, sign: false });
    data.append(FP8x23 { mag: 218103808, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
