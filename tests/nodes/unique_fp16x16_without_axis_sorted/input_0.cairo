use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 18803, sign: false });
    data.append(FP16x16 { mag: 42065, sign: false });
    data.append(FP16x16 { mag: 133958, sign: false });
    data.append(FP16x16 { mag: 129359, sign: false });
    data.append(FP16x16 { mag: 76398, sign: false });
    data.append(FP16x16 { mag: 127485, sign: false });
    data.append(FP16x16 { mag: 41739, sign: false });
    data.append(FP16x16 { mag: 36606, sign: false });
    data.append(FP16x16 { mag: 99432, sign: false });
    data.append(FP16x16 { mag: 93731, sign: false });
    data.append(FP16x16 { mag: 40635, sign: false });
    data.append(FP16x16 { mag: 146044, sign: false });
    data.append(FP16x16 { mag: 45156, sign: false });
    data.append(FP16x16 { mag: 92658, sign: false });
    data.append(FP16x16 { mag: 101742, sign: false });
    data.append(FP16x16 { mag: 115036, sign: false });
    data.append(FP16x16 { mag: 121002, sign: false });
    data.append(FP16x16 { mag: 30248, sign: false });
    data.append(FP16x16 { mag: 80580, sign: false });
    data.append(FP16x16 { mag: 17017, sign: false });
    data.append(FP16x16 { mag: 15148, sign: false });
    data.append(FP16x16 { mag: 8024, sign: false });
    data.append(FP16x16 { mag: 151441, sign: false });
    data.append(FP16x16 { mag: 54931, sign: false });
    data.append(FP16x16 { mag: 177975, sign: false });
    data.append(FP16x16 { mag: 189528, sign: false });
    data.append(FP16x16 { mag: 179335, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
