use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 60767, sign: false });
    data.append(FP16x16 { mag: 61719, sign: false });
    data.append(FP16x16 { mag: 14321, sign: false });
    data.append(FP16x16 { mag: 48660, sign: false });
    data.append(FP16x16 { mag: 53016, sign: false });
    data.append(FP16x16 { mag: 54477, sign: false });
    data.append(FP16x16 { mag: 9349, sign: false });
    data.append(FP16x16 { mag: 52579, sign: false });
    data.append(FP16x16 { mag: 91823, sign: false });
    data.append(FP16x16 { mag: 93878, sign: false });
    data.append(FP16x16 { mag: 16370, sign: false });
    data.append(FP16x16 { mag: 64215, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
