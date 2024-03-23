use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 1984692, sign: false });
    data.append(FP8x23 { mag: 6048216, sign: false });
    data.append(FP8x23 { mag: 7415753, sign: true });
    data.append(FP8x23 { mag: 5261100, sign: true });
    data.append(FP8x23 { mag: 146870, sign: true });
    data.append(FP8x23 { mag: 6269863, sign: false });
    data.append(FP8x23 { mag: 7481818, sign: true });
    data.append(FP8x23 { mag: 5274608, sign: true });
    data.append(FP8x23 { mag: 1245296, sign: true });
    data.append(FP8x23 { mag: 6384082, sign: false });
    data.append(FP8x23 { mag: 6360993, sign: true });
    data.append(FP8x23 { mag: 5045434, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
