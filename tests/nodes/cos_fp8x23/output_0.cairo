use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorSub};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 8203389, sign: true });
    data.append(FP8x23 { mag: 6673244, sign: true });
    data.append(FP8x23 { mag: 3338956, sign: true });
    data.append(FP8x23 { mag: 1649907, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
