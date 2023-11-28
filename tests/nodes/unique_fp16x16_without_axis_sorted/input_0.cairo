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
    data.append(FP16x16 { mag: 32314, sign: false });
    data.append(FP16x16 { mag: 116332, sign: false });
    data.append(FP16x16 { mag: 162730, sign: false });
    data.append(FP16x16 { mag: 153570, sign: false });
    data.append(FP16x16 { mag: 77465, sign: false });
    data.append(FP16x16 { mag: 51947, sign: false });
    data.append(FP16x16 { mag: 2997, sign: false });
    data.append(FP16x16 { mag: 59590, sign: false });
    data.append(FP16x16 { mag: 168764, sign: false });
    data.append(FP16x16 { mag: 188449, sign: false });
    data.append(FP16x16 { mag: 185811, sign: false });
    data.append(FP16x16 { mag: 148882, sign: false });
    data.append(FP16x16 { mag: 139498, sign: false });
    data.append(FP16x16 { mag: 73709, sign: false });
    data.append(FP16x16 { mag: 184393, sign: false });
    data.append(FP16x16 { mag: 70132, sign: false });
    data.append(FP16x16 { mag: 23780, sign: false });
    data.append(FP16x16 { mag: 119286, sign: false });
    data.append(FP16x16 { mag: 63510, sign: false });
    data.append(FP16x16 { mag: 11925, sign: false });
    data.append(FP16x16 { mag: 80588, sign: false });
    data.append(FP16x16 { mag: 161148, sign: false });
    data.append(FP16x16 { mag: 47213, sign: false });
    data.append(FP16x16 { mag: 45615, sign: false });
    data.append(FP16x16 { mag: 160150, sign: false });
    data.append(FP16x16 { mag: 91119, sign: false });
    data.append(FP16x16 { mag: 75209, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
