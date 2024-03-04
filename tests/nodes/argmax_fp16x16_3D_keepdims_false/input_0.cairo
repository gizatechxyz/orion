use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 1572864, sign: false });
    data.append(FP16x16 { mag: 2424832, sign: true });
    data.append(FP16x16 { mag: 6422528, sign: false });
    data.append(FP16x16 { mag: 5242880, sign: false });
    data.append(FP16x16 { mag: 1703936, sign: true });
    data.append(FP16x16 { mag: 6291456, sign: false });
    data.append(FP16x16 { mag: 589824, sign: true });
    data.append(FP16x16 { mag: 7733248, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
