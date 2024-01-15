use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorSub};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 119090, sign: false });
    data.append(FP16x16 { mag: 122540, sign: false });
    data.append(FP16x16 { mag: 89241, sign: false });
    data.append(FP16x16 { mag: 101578, sign: false });
    data.append(FP16x16 { mag: 181607, sign: false });
    data.append(FP16x16 { mag: 212926, sign: false });
    data.append(FP16x16 { mag: 141595, sign: false });
    data.append(FP16x16 { mag: 153059, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
