use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 78056, sign: true });
    data.append(FP16x16 { mag: 31456, sign: true });
    data.append(FP16x16 { mag: 172639, sign: true });
    data.append(FP16x16 { mag: 78597, sign: false });
    data.append(FP16x16 { mag: 62154, sign: false });
    data.append(FP16x16 { mag: 171656, sign: false });
    data.append(FP16x16 { mag: 157535, sign: false });
    data.append(FP16x16 { mag: 49284, sign: false });
    data.append(FP16x16 { mag: 97008, sign: false });
    data.append(FP16x16 { mag: 123759, sign: true });
    data.append(FP16x16 { mag: 190267, sign: false });
    data.append(FP16x16 { mag: 107363, sign: false });
    data.append(FP16x16 { mag: 7956, sign: true });
    data.append(FP16x16 { mag: 68542, sign: false });
    data.append(FP16x16 { mag: 116678, sign: false });
    data.append(FP16x16 { mag: 85597, sign: false });
    data.append(FP16x16 { mag: 19210, sign: true });
    data.append(FP16x16 { mag: 99774, sign: false });
    data.append(FP16x16 { mag: 173484, sign: false });
    data.append(FP16x16 { mag: 127017, sign: true });
    data.append(FP16x16 { mag: 83696, sign: false });
    data.append(FP16x16 { mag: 16087, sign: true });
    data.append(FP16x16 { mag: 80426, sign: false });
    data.append(FP16x16 { mag: 187986, sign: false });
    data.append(FP16x16 { mag: 45262, sign: true });
    data.append(FP16x16 { mag: 46955, sign: false });
    data.append(FP16x16 { mag: 38631, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
