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
    data.append(FP16x16 { mag: 3342336, sign: false });
    data.append(FP16x16 { mag: 7602176, sign: true });
    data.append(FP16x16 { mag: 3211264, sign: false });
    data.append(FP16x16 { mag: 2490368, sign: true });
    data.append(FP16x16 { mag: 589824, sign: false });
    data.append(FP16x16 { mag: 5570560, sign: true });
    data.append(FP16x16 { mag: 1900544, sign: true });
    data.append(FP16x16 { mag: 5898240, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
