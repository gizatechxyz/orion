use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(2);
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 486254, sign: false });
    data.append(FP16x16 { mag: 487133, sign: false });
    data.append(FP16x16 { mag: 112122, sign: false });
    data.append(FP16x16 { mag: 485208, sign: false });
    data.append(FP16x16 { mag: 565927, sign: false });
    data.append(FP16x16 { mag: 590441, sign: false });
    data.append(FP16x16 { mag: 73227, sign: false });
    data.append(FP16x16 { mag: 201392, sign: false });
    data.append(FP16x16 { mag: 342573, sign: false });
    data.append(FP16x16 { mag: 245684, sign: false });
    data.append(FP16x16 { mag: 368847, sign: false });
    data.append(FP16x16 { mag: 134871, sign: false });
    data.append(FP16x16 { mag: 449533, sign: false });
    data.append(FP16x16 { mag: 284826, sign: false });
    data.append(FP16x16 { mag: 234950, sign: false });
    data.append(FP16x16 { mag: 515285, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
