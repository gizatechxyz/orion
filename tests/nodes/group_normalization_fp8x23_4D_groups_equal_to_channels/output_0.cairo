use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 17999664, sign: true });
    data.append(FP8x23 { mag: 9775580, sign: false });
    data.append(FP8x23 { mag: 8947998, sign: true });
    data.append(FP8x23 { mag: 5250632, sign: true });
    data.append(FP8x23 { mag: 21183, sign: false });
    data.append(FP8x23 { mag: 8546804, sign: false });
    data.append(FP8x23 { mag: 10708254, sign: false });
    data.append(FP8x23 { mag: 1351987, sign: true });
    data.append(FP8x23 { mag: 391354, sign: false });
    data.append(FP8x23 { mag: 14376598, sign: true });
    data.append(FP8x23 { mag: 7086189, sign: false });
    data.append(FP8x23 { mag: 15523657, sign: true });
    data.append(FP8x23 { mag: 10403368, sign: false });
    data.append(FP8x23 { mag: 7338979, sign: false });
    data.append(FP8x23 { mag: 4204091, sign: false });
    data.append(FP8x23 { mag: 4022184, sign: true });
    data.append(FP8x23 { mag: 21697788, sign: true });
    data.append(FP8x23 { mag: 1148774, sign: true });
    data.append(FP8x23 { mag: 3716845, sign: false });
    data.append(FP8x23 { mag: 3292994, sign: true });
    data.append(FP8x23 { mag: 5418566, sign: false });
    data.append(FP8x23 { mag: 347196, sign: false });
    data.append(FP8x23 { mag: 12088816, sign: false });
    data.append(FP8x23 { mag: 69675, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
