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
    data.append(FP16x16 { mag: 131841, sign: false });
    data.append(FP16x16 { mag: 177122, sign: false });
    data.append(FP16x16 { mag: 8001, sign: true });
    data.append(FP16x16 { mag: 196243, sign: false });
    data.append(FP16x16 { mag: 92229, sign: true });
    data.append(FP16x16 { mag: 51294, sign: false });
    data.append(FP16x16 { mag: 106419, sign: false });
    data.append(FP16x16 { mag: 174717, sign: false });
    data.append(FP16x16 { mag: 140928, sign: true });
    data.append(FP16x16 { mag: 11465, sign: true });
    data.append(FP16x16 { mag: 140627, sign: true });
    data.append(FP16x16 { mag: 176437, sign: false });
    data.append(FP16x16 { mag: 104537, sign: false });
    data.append(FP16x16 { mag: 127197, sign: false });
    data.append(FP16x16 { mag: 7148, sign: true });
    data.append(FP16x16 { mag: 16541, sign: true });
    data.append(FP16x16 { mag: 60120, sign: false });
    data.append(FP16x16 { mag: 192460, sign: true });
    data.append(FP16x16 { mag: 152480, sign: true });
    data.append(FP16x16 { mag: 118232, sign: true });
    data.append(FP16x16 { mag: 55096, sign: true });
    data.append(FP16x16 { mag: 174569, sign: true });
    data.append(FP16x16 { mag: 49745, sign: true });
    data.append(FP16x16 { mag: 36189, sign: true });
    data.append(FP16x16 { mag: 26667, sign: false });
    data.append(FP16x16 { mag: 89957, sign: true });
    data.append(FP16x16 { mag: 112614, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
