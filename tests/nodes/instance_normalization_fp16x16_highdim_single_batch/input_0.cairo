use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(1);
    shape.append(2);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 125788, sign: false });
    data.append(FP16x16 { mag: 29301, sign: false });
    data.append(FP16x16 { mag: 50693, sign: false });
    data.append(FP16x16 { mag: 43193, sign: false });
    data.append(FP16x16 { mag: 61038, sign: false });
    data.append(FP16x16 { mag: 49566, sign: true });
    data.append(FP16x16 { mag: 46456, sign: true });
    data.append(FP16x16 { mag: 7004, sign: false });
    data.append(FP16x16 { mag: 70178, sign: true });
    data.append(FP16x16 { mag: 91824, sign: true });
    data.append(FP16x16 { mag: 147137, sign: false });
    data.append(FP16x16 { mag: 2698, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
