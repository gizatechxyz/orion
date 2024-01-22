use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(1);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 5505024, sign: true });
    data.append(FP16x16 { mag: 1441792, sign: false });
    data.append(FP16x16 { mag: 983040, sign: false });
    data.append(FP16x16 { mag: 8060928, sign: false });
    data.append(FP16x16 { mag: 2818048, sign: false });
    data.append(FP16x16 { mag: 4521984, sign: true });
    data.append(FP16x16 { mag: 3080192, sign: false });
    data.append(FP16x16 { mag: 5767168, sign: false });
    data.append(FP16x16 { mag: 4849664, sign: true });
    data.append(FP16x16 { mag: 1900544, sign: true });
    data.append(FP16x16 { mag: 1310720, sign: true });
    data.append(FP16x16 { mag: 655360, sign: true });
    data.append(FP16x16 { mag: 2883584, sign: false });
    data.append(FP16x16 { mag: 4456448, sign: false });
    data.append(FP16x16 { mag: 851968, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
