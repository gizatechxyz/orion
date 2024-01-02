use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 3211264, sign: false });
    data.append(FP16x16 { mag: 7602176, sign: true });
    data.append(FP16x16 { mag: 3932160, sign: true });
    data.append(FP16x16 { mag: 2686976, sign: true });
    data.append(FP16x16 { mag: 5767168, sign: true });
    data.append(FP16x16 { mag: 6488064, sign: false });
    data.append(FP16x16 { mag: 7340032, sign: false });
    data.append(FP16x16 { mag: 983040, sign: true });
    data.append(FP16x16 { mag: 4259840, sign: false });
    data.append(FP16x16 { mag: 7995392, sign: false });
    data.append(FP16x16 { mag: 851968, sign: false });
    data.append(FP16x16 { mag: 3735552, sign: true });
    data.append(FP16x16 { mag: 3997696, sign: false });
    data.append(FP16x16 { mag: 6422528, sign: true });
    data.append(FP16x16 { mag: 3014656, sign: false });
    data.append(FP16x16 { mag: 2359296, sign: false });
    data.append(FP16x16 { mag: 7208960, sign: false });
    data.append(FP16x16 { mag: 1310720, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
