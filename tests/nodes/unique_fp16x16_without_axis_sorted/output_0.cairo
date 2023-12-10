use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(27);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 18800, sign: false });
    data.append(FP16x16 { mag: 42080, sign: false });
    data.append(FP16x16 { mag: 134016, sign: false });
    data.append(FP16x16 { mag: 129344, sign: false });
    data.append(FP16x16 { mag: 76416, sign: false });
    data.append(FP16x16 { mag: 127488, sign: false });
    data.append(FP16x16 { mag: 41728, sign: false });
    data.append(FP16x16 { mag: 36608, sign: false });
    data.append(FP16x16 { mag: 99456, sign: false });
    data.append(FP16x16 { mag: 93760, sign: false });
    data.append(FP16x16 { mag: 40640, sign: false });
    data.append(FP16x16 { mag: 146048, sign: false });
    data.append(FP16x16 { mag: 45152, sign: false });
    data.append(FP16x16 { mag: 92672, sign: false });
    data.append(FP16x16 { mag: 101760, sign: false });
    data.append(FP16x16 { mag: 115008, sign: false });
    data.append(FP16x16 { mag: 121024, sign: false });
    data.append(FP16x16 { mag: 30256, sign: false });
    data.append(FP16x16 { mag: 80576, sign: false });
    data.append(FP16x16 { mag: 17024, sign: false });
    data.append(FP16x16 { mag: 15152, sign: false });
    data.append(FP16x16 { mag: 8024, sign: false });
    data.append(FP16x16 { mag: 151424, sign: false });
    data.append(FP16x16 { mag: 54944, sign: false });
    data.append(FP16x16 { mag: 177920, sign: false });
    data.append(FP16x16 { mag: 189568, sign: false });
    data.append(FP16x16 { mag: 179328, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
