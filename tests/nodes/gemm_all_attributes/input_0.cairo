use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 6181, sign: false });
    data.append(FP16x16 { mag: 64225, sign: false });
    data.append(FP16x16 { mag: 49893, sign: false });
    data.append(FP16x16 { mag: 27262, sign: false });
    data.append(FP16x16 { mag: 59085, sign: false });
    data.append(FP16x16 { mag: 11503, sign: false });
    data.append(FP16x16 { mag: 27421, sign: false });
    data.append(FP16x16 { mag: 1528, sign: false });
    data.append(FP16x16 { mag: 3846, sign: false });
    data.append(FP16x16 { mag: 45763, sign: false });
    data.append(FP16x16 { mag: 23273, sign: false });
    data.append(FP16x16 { mag: 2087, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
