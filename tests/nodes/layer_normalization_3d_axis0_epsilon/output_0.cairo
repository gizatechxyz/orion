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
    data.append(FP16x16 { mag: 42429, sign: false });
    data.append(FP16x16 { mag: 49634, sign: true });
    data.append(FP16x16 { mag: 77948, sign: true });
    data.append(FP16x16 { mag: 93325, sign: true });
    data.append(FP16x16 { mag: 84777, sign: true });
    data.append(FP16x16 { mag: 143407, sign: true });
    data.append(FP16x16 { mag: 133737, sign: true });
    data.append(FP16x16 { mag: 102353, sign: true });
    data.append(FP16x16 { mag: 46642, sign: true });
    data.append(FP16x16 { mag: 130431, sign: false });
    data.append(FP16x16 { mag: 316659, sign: false });
    data.append(FP16x16 { mag: 76609, sign: true });
    data.append(FP16x16 { mag: 15411, sign: true });
    data.append(FP16x16 { mag: 103656, sign: true });
    data.append(FP16x16 { mag: 14405, sign: true });
    data.append(FP16x16 { mag: 68539, sign: true });
    data.append(FP16x16 { mag: 2850, sign: false });
    data.append(FP16x16 { mag: 11550, sign: false });
    data.append(FP16x16 { mag: 50032, sign: true });
    data.append(FP16x16 { mag: 145784, sign: true });
    data.append(FP16x16 { mag: 35540, sign: true });
    data.append(FP16x16 { mag: 15615, sign: false });
    data.append(FP16x16 { mag: 59548, sign: false });
    data.append(FP16x16 { mag: 66808, sign: false });
    data.append(FP16x16 { mag: 132508, sign: true });
    data.append(FP16x16 { mag: 37408, sign: false });
    data.append(FP16x16 { mag: 116363, sign: false });
    data.append(FP16x16 { mag: 60021, sign: true });
    data.append(FP16x16 { mag: 11310, sign: false });
    data.append(FP16x16 { mag: 56233, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
