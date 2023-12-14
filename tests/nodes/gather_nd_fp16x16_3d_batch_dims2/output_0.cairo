use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 196608, sign: false });
    data.append(FP16x16 { mag: 524288, sign: false });
    data.append(FP16x16 { mag: 917504, sign: false });
    data.append(FP16x16 { mag: 1310720, sign: false });
    data.append(FP16x16 { mag: 1703936, sign: false });
    data.append(FP16x16 { mag: 2097152, sign: false });
    data.append(FP16x16 { mag: 2424832, sign: false });
    data.append(FP16x16 { mag: 2752512, sign: false });
    data.append(FP16x16 { mag: 3145728, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
