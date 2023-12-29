use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 5898240, sign: false });
    data.append(FP16x16 { mag: 6356992, sign: false });
    data.append(FP16x16 { mag: 3276800, sign: true });
    data.append(FP16x16 { mag: 7536640, sign: true });
    data.append(FP16x16 { mag: 3014656, sign: true });
    data.append(FP16x16 { mag: 8257536, sign: false });
    data.append(FP16x16 { mag: 4456448, sign: false });
    data.append(FP16x16 { mag: 655360, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
