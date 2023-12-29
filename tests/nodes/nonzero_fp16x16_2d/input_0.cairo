use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 7536640, sign: true });
    data.append(FP16x16 { mag: 3670016, sign: true });
    data.append(FP16x16 { mag: 1572864, sign: false });
    data.append(FP16x16 { mag: 2818048, sign: true });
    data.append(FP16x16 { mag: 2162688, sign: true });
    data.append(FP16x16 { mag: 2031616, sign: true });
    data.append(FP16x16 { mag: 524288, sign: true });
    data.append(FP16x16 { mag: 5963776, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
