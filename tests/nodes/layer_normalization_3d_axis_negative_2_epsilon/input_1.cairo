use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 69151, sign: false });
    data.append(FP16x16 { mag: 49647, sign: true });
    data.append(FP16x16 { mag: 25308, sign: false });
    data.append(FP16x16 { mag: 49767, sign: true });
    data.append(FP16x16 { mag: 17913, sign: false });
    data.append(FP16x16 { mag: 10090, sign: false });
    data.append(FP16x16 { mag: 4364, sign: true });
    data.append(FP16x16 { mag: 41827, sign: false });
    data.append(FP16x16 { mag: 12848, sign: true });
    data.append(FP16x16 { mag: 137417, sign: false });
    data.append(FP16x16 { mag: 70720, sign: true });
    data.append(FP16x16 { mag: 76109, sign: false });
    data.append(FP16x16 { mag: 66590, sign: true });
    data.append(FP16x16 { mag: 50658, sign: false });
    data.append(FP16x16 { mag: 107949, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
