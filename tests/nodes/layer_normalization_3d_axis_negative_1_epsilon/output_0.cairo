use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 33712, sign: true });
    data.append(FP16x16 { mag: 24229, sign: true });
    data.append(FP16x16 { mag: 93572, sign: true });
    data.append(FP16x16 { mag: 36316, sign: true });
    data.append(FP16x16 { mag: 55471, sign: true });
    data.append(FP16x16 { mag: 25943, sign: true });
    data.append(FP16x16 { mag: 24906, sign: true });
    data.append(FP16x16 { mag: 13890, sign: true });
    data.append(FP16x16 { mag: 41064, sign: true });
    data.append(FP16x16 { mag: 80454, sign: true });
    data.append(FP16x16 { mag: 55145, sign: true });
    data.append(FP16x16 { mag: 86832, sign: true });
    data.append(FP16x16 { mag: 21959, sign: true });
    data.append(FP16x16 { mag: 40289, sign: true });
    data.append(FP16x16 { mag: 55852, sign: true });
    data.append(FP16x16 { mag: 49742, sign: true });
    data.append(FP16x16 { mag: 3830, sign: true });
    data.append(FP16x16 { mag: 28080, sign: false });
    data.append(FP16x16 { mag: 38856, sign: true });
    data.append(FP16x16 { mag: 45753, sign: true });
    data.append(FP16x16 { mag: 33307, sign: true });
    data.append(FP16x16 { mag: 11580, sign: false });
    data.append(FP16x16 { mag: 36059, sign: false });
    data.append(FP16x16 { mag: 38461, sign: true });
    data.append(FP16x16 { mag: 76347, sign: true });
    data.append(FP16x16 { mag: 38152, sign: true });
    data.append(FP16x16 { mag: 13646, sign: true });
    data.append(FP16x16 { mag: 16354, sign: true });
    data.append(FP16x16 { mag: 43407, sign: true });
    data.append(FP16x16 { mag: 35849, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
