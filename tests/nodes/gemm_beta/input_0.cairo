use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(7);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 24460, sign: false });
    data.append(FP16x16 { mag: 18819, sign: false });
    data.append(FP16x16 { mag: 5981, sign: false });
    data.append(FP16x16 { mag: 57425, sign: false });
    data.append(FP16x16 { mag: 3433, sign: false });
    data.append(FP16x16 { mag: 51302, sign: false });
    data.append(FP16x16 { mag: 30317, sign: false });
    data.append(FP16x16 { mag: 51496, sign: false });
    data.append(FP16x16 { mag: 49111, sign: false });
    data.append(FP16x16 { mag: 60422, sign: false });
    data.append(FP16x16 { mag: 62691, sign: false });
    data.append(FP16x16 { mag: 49763, sign: false });
    data.append(FP16x16 { mag: 54999, sign: false });
    data.append(FP16x16 { mag: 31795, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
