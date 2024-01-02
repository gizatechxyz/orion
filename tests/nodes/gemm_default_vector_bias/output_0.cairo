use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 184581, sign: false });
    data.append(FP16x16 { mag: 159946, sign: false });
    data.append(FP16x16 { mag: 169795, sign: false });
    data.append(FP16x16 { mag: 137561, sign: false });
    data.append(FP16x16 { mag: 153754, sign: false });
    data.append(FP16x16 { mag: 150739, sign: false });
    data.append(FP16x16 { mag: 87691, sign: false });
    data.append(FP16x16 { mag: 106281, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
