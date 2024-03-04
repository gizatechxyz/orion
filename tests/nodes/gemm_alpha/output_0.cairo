use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 31676, sign: false });
    data.append(FP16x16 { mag: 54792, sign: false });
    data.append(FP16x16 { mag: 76833, sign: false });
    data.append(FP16x16 { mag: 50845, sign: false });
    data.append(FP16x16 { mag: 14333, sign: false });
    data.append(FP16x16 { mag: 28040, sign: false });
    data.append(FP16x16 { mag: 41871, sign: false });
    data.append(FP16x16 { mag: 21330, sign: false });
    data.append(FP16x16 { mag: 31873, sign: false });
    data.append(FP16x16 { mag: 48155, sign: false });
    data.append(FP16x16 { mag: 73048, sign: false });
    data.append(FP16x16 { mag: 52353, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
