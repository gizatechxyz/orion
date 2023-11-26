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
    data.append(FP16x16 { mag: 167983, sign: false });
    data.append(FP16x16 { mag: 103145, sign: false });
    data.append(FP16x16 { mag: 73690, sign: false });
    data.append(FP16x16 { mag: 187012, sign: false });
    data.append(FP16x16 { mag: 81311, sign: false });
    data.append(FP16x16 { mag: 7340, sign: false });
    data.append(FP16x16 { mag: 83755, sign: false });
    data.append(FP16x16 { mag: 97056, sign: false });
    data.append(FP16x16 { mag: 96142, sign: false });
    data.append(FP16x16 { mag: 80995, sign: false });
    data.append(FP16x16 { mag: 10897, sign: false });
    data.append(FP16x16 { mag: 56112, sign: false });
    data.append(FP16x16 { mag: 85459, sign: false });
    data.append(FP16x16 { mag: 82620, sign: false });
    data.append(FP16x16 { mag: 93533, sign: false });
    data.append(FP16x16 { mag: 146364, sign: false });
    data.append(FP16x16 { mag: 23371, sign: false });
    data.append(FP16x16 { mag: 175749, sign: false });
    data.append(FP16x16 { mag: 122942, sign: false });
    data.append(FP16x16 { mag: 58181, sign: false });
    data.append(FP16x16 { mag: 65506, sign: false });
    data.append(FP16x16 { mag: 167127, sign: false });
    data.append(FP16x16 { mag: 189422, sign: false });
    data.append(FP16x16 { mag: 52812, sign: false });
    data.append(FP16x16 { mag: 14410, sign: false });
    data.append(FP16x16 { mag: 70276, sign: false });
    data.append(FP16x16 { mag: 62764, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
