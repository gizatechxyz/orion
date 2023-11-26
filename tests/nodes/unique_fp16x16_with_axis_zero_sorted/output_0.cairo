use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 87936, sign: true });
    data.append(FP16x16 { mag: 9320, sign: false });
    data.append(FP16x16 { mag: 66240, sign: true });
    data.append(FP16x16 { mag: 156288, sign: false });
    data.append(FP16x16 { mag: 78016, sign: false });
    data.append(FP16x16 { mag: 184064, sign: false });
    data.append(FP16x16 { mag: 28976, sign: false });
    data.append(FP16x16 { mag: 47168, sign: true });
    data.append(FP16x16 { mag: 143872, sign: false });
    data.append(FP16x16 { mag: 17232, sign: false });
    data.append(FP16x16 { mag: 168192, sign: false });
    data.append(FP16x16 { mag: 140672, sign: false });
    data.append(FP16x16 { mag: 196352, sign: false });
    data.append(FP16x16 { mag: 126336, sign: false });
    data.append(FP16x16 { mag: 190464, sign: true });
    data.append(FP16x16 { mag: 196096, sign: false });
    data.append(FP16x16 { mag: 129920, sign: true });
    data.append(FP16x16 { mag: 47488, sign: true });
    data.append(FP16x16 { mag: 173824, sign: true });
    data.append(FP16x16 { mag: 179328, sign: false });
    data.append(FP16x16 { mag: 151936, sign: true });
    data.append(FP16x16 { mag: 195200, sign: false });
    data.append(FP16x16 { mag: 71296, sign: false });
    data.append(FP16x16 { mag: 71488, sign: false });
    data.append(FP16x16 { mag: 98880, sign: false });
    data.append(FP16x16 { mag: 18768, sign: true });
    data.append(FP16x16 { mag: 194048, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
