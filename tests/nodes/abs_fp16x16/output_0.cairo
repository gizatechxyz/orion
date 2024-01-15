use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 3670016, sign: false });
    data.append(FP16x16 { mag: 7208960, sign: false });
    data.append(FP16x16 { mag: 3014656, sign: false });
    data.append(FP16x16 { mag: 3932160, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
