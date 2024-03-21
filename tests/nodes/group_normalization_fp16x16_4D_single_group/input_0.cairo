use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 52467, sign: true });
    data.append(FP16x16 { mag: 17332, sign: false });
    data.append(FP16x16 { mag: 111758, sign: false });
    data.append(FP16x16 { mag: 39486, sign: false });
    data.append(FP16x16 { mag: 115260, sign: false });
    data.append(FP16x16 { mag: 19805, sign: true });
    data.append(FP16x16 { mag: 64934, sign: false });
    data.append(FP16x16 { mag: 15961, sign: false });
    data.append(FP16x16 { mag: 89303, sign: true });
    data.append(FP16x16 { mag: 41481, sign: true });
    data.append(FP16x16 { mag: 15148, sign: false });
    data.append(FP16x16 { mag: 72183, sign: false });
    data.append(FP16x16 { mag: 26969, sign: false });
    data.append(FP16x16 { mag: 76254, sign: true });
    data.append(FP16x16 { mag: 120941, sign: false });
    data.append(FP16x16 { mag: 7636, sign: false });
    data.append(FP16x16 { mag: 171308, sign: false });
    data.append(FP16x16 { mag: 35654, sign: true });
    data.append(FP16x16 { mag: 40285, sign: true });
    data.append(FP16x16 { mag: 57045, sign: false });
    data.append(FP16x16 { mag: 85827, sign: true });
    data.append(FP16x16 { mag: 27805, sign: false });
    data.append(FP16x16 { mag: 61115, sign: false });
    data.append(FP16x16 { mag: 21447, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
