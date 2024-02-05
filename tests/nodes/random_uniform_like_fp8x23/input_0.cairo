use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 6985047, sign: true });
    data.append(FP8x23 { mag: 18908887, sign: false });
    data.append(FP8x23 { mag: 24337672, sign: true });
    data.append(FP8x23 { mag: 2420381, sign: false });
    data.append(FP8x23 { mag: 1071211, sign: true });
    data.append(FP8x23 { mag: 20033413, sign: true });
    data.append(FP8x23 { mag: 217485, sign: true });
    data.append(FP8x23 { mag: 4968906, sign: false });
    data.append(FP8x23 { mag: 5503174, sign: false });
    data.append(FP8x23 { mag: 4333577, sign: false });
    data.append(FP8x23 { mag: 16341821, sign: true });
    data.append(FP8x23 { mag: 18925428, sign: true });
    data.append(FP8x23 { mag: 17251664, sign: false });
    data.append(FP8x23 { mag: 23832813, sign: false });
    data.append(FP8x23 { mag: 3968519, sign: false });
    data.append(FP8x23 { mag: 22692691, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
