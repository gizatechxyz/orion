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
    data.append(FP16x16 { mag: 1966080, sign: false });
    data.append(FP16x16 { mag: 4653056, sign: false });
    data.append(FP16x16 { mag: 1835008, sign: false });
    data.append(FP16x16 { mag: 5767168, sign: true });
    data.append(FP16x16 { mag: 7929856, sign: true });
    data.append(FP16x16 { mag: 2162688, sign: true });
    data.append(FP16x16 { mag: 4718592, sign: false });
    data.append(FP16x16 { mag: 196608, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
