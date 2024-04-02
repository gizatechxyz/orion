use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 34237, sign: true });
    data.append(FP16x16 { mag: 91593, sign: true });
    data.append(FP16x16 { mag: 86945, sign: true });
    data.append(FP16x16 { mag: 119327, sign: false });
    data.append(FP16x16 { mag: 84721, sign: true });
    data.append(FP16x16 { mag: 91369, sign: false });
    data.append(FP16x16 { mag: 112976, sign: false });
    data.append(FP16x16 { mag: 91367, sign: false });
    data.append(FP16x16 { mag: 1404, sign: false });
    data.append(FP16x16 { mag: 90190, sign: true });
    data.append(FP16x16 { mag: 53580, sign: true });
    data.append(FP16x16 { mag: 47253, sign: true });
    data.append(FP16x16 { mag: 109695, sign: true });
    data.append(FP16x16 { mag: 152377, sign: false });
    data.append(FP16x16 { mag: 33357, sign: false });
    data.append(FP16x16 { mag: 41154, sign: true });
    data.append(FP16x16 { mag: 71017, sign: true });
    data.append(FP16x16 { mag: 72489, sign: false });
    data.append(FP16x16 { mag: 12155, sign: false });
    data.append(FP16x16 { mag: 69555, sign: true });
    data.append(FP16x16 { mag: 37131, sign: false });
    data.append(FP16x16 { mag: 6528, sign: false });
    data.append(FP16x16 { mag: 49257, sign: true });
    data.append(FP16x16 { mag: 20260, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
