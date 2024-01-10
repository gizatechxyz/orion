use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 1966080, sign: false });
    data.append(FP16x16 { mag: 8257536, sign: false });
    data.append(FP16x16 { mag: 7471104, sign: true });
    data.append(FP16x16 { mag: 4849664, sign: true });
    data.append(FP16x16 { mag: 3407872, sign: false });
    data.append(FP16x16 { mag: 3014656, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
