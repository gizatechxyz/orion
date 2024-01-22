use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(6);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 24313, sign: false });
    data.append(FP16x16 { mag: 15462, sign: false });
    data.append(FP16x16 { mag: 40752, sign: false });
    data.append(FP16x16 { mag: 22033, sign: false });
    data.append(FP16x16 { mag: 30592, sign: false });
    data.append(FP16x16 { mag: 46888, sign: false });
    data.append(FP16x16 { mag: 16081, sign: false });
    data.append(FP16x16 { mag: 12404, sign: false });
    data.append(FP16x16 { mag: 27033, sign: false });
    data.append(FP16x16 { mag: 10395, sign: false });
    data.append(FP16x16 { mag: 12042, sign: false });
    data.append(FP16x16 { mag: 35690, sign: false });
    data.append(FP16x16 { mag: 7033, sign: false });
    data.append(FP16x16 { mag: 606, sign: false });
    data.append(FP16x16 { mag: 2994, sign: false });
    data.append(FP16x16 { mag: 20651, sign: false });
    data.append(FP16x16 { mag: 46099, sign: false });
    data.append(FP16x16 { mag: 5768, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
