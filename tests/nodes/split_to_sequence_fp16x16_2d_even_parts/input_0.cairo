use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 851968, sign: true });
    data.append(FP16x16 { mag: 8126464, sign: true });
    data.append(FP16x16 { mag: 2621440, sign: true });
    data.append(FP16x16 { mag: 7143424, sign: false });
    data.append(FP16x16 { mag: 3735552, sign: false });
    data.append(FP16x16 { mag: 8060928, sign: true });
    data.append(FP16x16 { mag: 3932160, sign: false });
    data.append(FP16x16 { mag: 3473408, sign: false });
    data.append(FP16x16 { mag: 5373952, sign: false });
    data.append(FP16x16 { mag: 6750208, sign: false });
    data.append(FP16x16 { mag: 6029312, sign: false });
    data.append(FP16x16 { mag: 6029312, sign: true });
    data.append(FP16x16 { mag: 7536640, sign: false });
    data.append(FP16x16 { mag: 7667712, sign: false });
    data.append(FP16x16 { mag: 6291456, sign: true });
    data.append(FP16x16 { mag: 655360, sign: false });
    data.append(FP16x16 { mag: 6356992, sign: false });
    data.append(FP16x16 { mag: 4849664, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
