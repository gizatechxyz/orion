use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 8094, sign: false });
    data.append(FP16x16 { mag: 14526, sign: false });
    data.append(FP16x16 { mag: 74969, sign: false });
    data.append(FP16x16 { mag: 6451, sign: true });
    data.append(FP16x16 { mag: 39284, sign: false });
    data.append(FP16x16 { mag: 80923, sign: true });
    data.append(FP16x16 { mag: 44892, sign: true });
    data.append(FP16x16 { mag: 28213, sign: false });
    data.append(FP16x16 { mag: 60140, sign: true });
    data.append(FP16x16 { mag: 12295, sign: true });
    data.append(FP16x16 { mag: 58313, sign: false });
    data.append(FP16x16 { mag: 123683, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
