use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 88819, sign: false });
    data.append(FP16x16 { mag: 70034, sign: false });
    data.append(FP16x16 { mag: 64340, sign: false });
    data.append(FP16x16 { mag: 84362, sign: false });
    data.append(FP16x16 { mag: 112807, sign: false });
    data.append(FP16x16 { mag: 142888, sign: false });
    data.append(FP16x16 { mag: 132618, sign: false });
    data.append(FP16x16 { mag: 122405, sign: false });
    data.append(FP16x16 { mag: 139852, sign: false });
    data.append(FP16x16 { mag: 128789, sign: false });
    data.append(FP16x16 { mag: 121667, sign: false });
    data.append(FP16x16 { mag: 153539, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
