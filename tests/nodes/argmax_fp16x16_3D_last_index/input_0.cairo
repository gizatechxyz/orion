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
    data.append(FP16x16 { mag: 6291456, sign: true });
    data.append(FP16x16 { mag: 7077888, sign: true });
    data.append(FP16x16 { mag: 1966080, sign: false });
    data.append(FP16x16 { mag: 4063232, sign: false });
    data.append(FP16x16 { mag: 5439488, sign: false });
    data.append(FP16x16 { mag: 1114112, sign: true });
    data.append(FP16x16 { mag: 1441792, sign: false });
    data.append(FP16x16 { mag: 458752, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
