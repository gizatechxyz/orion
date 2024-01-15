use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1065353216, sign: true });
    data.append(FP8x23 { mag: 352321536, sign: true });
    data.append(FP8x23 { mag: 486539264, sign: false });
    data.append(FP8x23 { mag: 301989888, sign: false });
    data.append(FP8x23 { mag: 847249408, sign: true });
    data.append(FP8x23 { mag: 696254464, sign: true });
    data.append(FP8x23 { mag: 545259520, sign: true });
    data.append(FP8x23 { mag: 8388608, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
