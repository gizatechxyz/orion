use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 75652487, sign: false });
    data.append(FP8x23 { mag: 84041095, sign: false });
    data.append(FP8x23 { mag: 92429703, sign: false });
    data.append(FP8x23 { mag: 100818311, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
