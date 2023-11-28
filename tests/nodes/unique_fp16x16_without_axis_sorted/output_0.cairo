use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(27);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 32320, sign: false });
    data.append(FP16x16 { mag: 116352, sign: false });
    data.append(FP16x16 { mag: 162688, sign: false });
    data.append(FP16x16 { mag: 153600, sign: false });
    data.append(FP16x16 { mag: 77440, sign: false });
    data.append(FP16x16 { mag: 51936, sign: false });
    data.append(FP16x16 { mag: 2998, sign: false });
    data.append(FP16x16 { mag: 59584, sign: false });
    data.append(FP16x16 { mag: 168704, sign: false });
    data.append(FP16x16 { mag: 188416, sign: false });
    data.append(FP16x16 { mag: 185856, sign: false });
    data.append(FP16x16 { mag: 148864, sign: false });
    data.append(FP16x16 { mag: 139520, sign: false });
    data.append(FP16x16 { mag: 73728, sign: false });
    data.append(FP16x16 { mag: 184448, sign: false });
    data.append(FP16x16 { mag: 70144, sign: false });
    data.append(FP16x16 { mag: 23776, sign: false });
    data.append(FP16x16 { mag: 119296, sign: false });
    data.append(FP16x16 { mag: 63520, sign: false });
    data.append(FP16x16 { mag: 11928, sign: false });
    data.append(FP16x16 { mag: 80576, sign: false });
    data.append(FP16x16 { mag: 161152, sign: false });
    data.append(FP16x16 { mag: 47200, sign: false });
    data.append(FP16x16 { mag: 45600, sign: false });
    data.append(FP16x16 { mag: 160128, sign: false });
    data.append(FP16x16 { mag: 91136, sign: false });
    data.append(FP16x16 { mag: 75200, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
