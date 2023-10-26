use array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_1() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(7);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 40337, sign: false });
    data.append(FP16x16 { mag: 29183, sign: false });
    data.append(FP16x16 { mag: 2662, sign: false });
    data.append(FP16x16 { mag: 26364, sign: false });
    data.append(FP16x16 { mag: 42934, sign: false });
    data.append(FP16x16 { mag: 65150, sign: false });
    data.append(FP16x16 { mag: 19395, sign: false });
    data.append(FP16x16 { mag: 39868, sign: false });
    data.append(FP16x16 { mag: 12023, sign: false });
    data.append(FP16x16 { mag: 28456, sign: false });
    data.append(FP16x16 { mag: 20310, sign: false });
    data.append(FP16x16 { mag: 33530, sign: false });
    data.append(FP16x16 { mag: 15549, sign: false });
    data.append(FP16x16 { mag: 37265, sign: false });
    data.append(FP16x16 { mag: 64596, sign: false });
    data.append(FP16x16 { mag: 58778, sign: false });
    data.append(FP16x16 { mag: 41122, sign: false });
    data.append(FP16x16 { mag: 29826, sign: false });
    data.append(FP16x16 { mag: 43424, sign: false });
    data.append(FP16x16 { mag: 47301, sign: false });
    data.append(FP16x16 { mag: 5420, sign: false });
    data.append(FP16x16 { mag: 54233, sign: false });
    data.append(FP16x16 { mag: 28313, sign: false });
    data.append(FP16x16 { mag: 12356, sign: false });
    data.append(FP16x16 { mag: 54540, sign: false });
    data.append(FP16x16 { mag: 42851, sign: false });
    data.append(FP16x16 { mag: 28457, sign: false });
    data.append(FP16x16 { mag: 16731, sign: false });
    TensorTrait::new(shape.span(), data.span())
}