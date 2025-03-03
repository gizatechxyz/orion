use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(1);
    shape.append(3);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 17957324, sign: true });
    data.append(FP8x23 { mag: 9284678, sign: true });
    data.append(FP8x23 { mag: 6177783, sign: true });
    data.append(FP8x23 { mag: 3454093, sign: true });
    data.append(FP8x23 { mag: 790232, sign: false });
    data.append(FP8x23 { mag: 7914969, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
