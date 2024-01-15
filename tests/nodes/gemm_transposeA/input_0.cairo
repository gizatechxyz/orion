use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(6);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 25712, sign: false });
    data.append(FP16x16 { mag: 33518, sign: false });
    data.append(FP16x16 { mag: 55900, sign: false });
    data.append(FP16x16 { mag: 17835, sign: false });
    data.append(FP16x16 { mag: 55991, sign: false });
    data.append(FP16x16 { mag: 49824, sign: false });
    data.append(FP16x16 { mag: 10885, sign: false });
    data.append(FP16x16 { mag: 46382, sign: false });
    data.append(FP16x16 { mag: 12037, sign: false });
    data.append(FP16x16 { mag: 54001, sign: false });
    data.append(FP16x16 { mag: 28270, sign: false });
    data.append(FP16x16 { mag: 53624, sign: false });
    data.append(FP16x16 { mag: 8973, sign: false });
    data.append(FP16x16 { mag: 57376, sign: false });
    data.append(FP16x16 { mag: 59443, sign: false });
    data.append(FP16x16 { mag: 20787, sign: false });
    data.append(FP16x16 { mag: 21410, sign: false });
    data.append(FP16x16 { mag: 30468, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
