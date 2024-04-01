use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use orion::numbers::{FixedTrait, FP16x16};

fn output_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(5);
    shape.append(4);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 90891, sign: true });
    data.append(FP16x16 { mag: 22909, sign: false });
    data.append(FP16x16 { mag: 42351, sign: true });
    data.append(FP16x16 { mag: 50940, sign: true });
    data.append(FP16x16 { mag: 28307, sign: true });
    data.append(FP16x16 { mag: 67021, sign: false });
    data.append(FP16x16 { mag: 89376, sign: false });
    data.append(FP16x16 { mag: 47022, sign: false });
    data.append(FP16x16 { mag: 77077, sign: false });
    data.append(FP16x16 { mag: 127415, sign: false });
    data.append(FP16x16 { mag: 39495, sign: true });
    data.append(FP16x16 { mag: 140364, sign: true });
    data.append(FP16x16 { mag: 210334, sign: false });
    data.append(FP16x16 { mag: 164609, sign: true });
    data.append(FP16x16 { mag: 106786, sign: false });
    data.append(FP16x16 { mag: 211953, sign: false });
    data.append(FP16x16 { mag: 4139, sign: false });
    data.append(FP16x16 { mag: 6330, sign: false });
    data.append(FP16x16 { mag: 110311, sign: false });
    data.append(FP16x16 { mag: 65149, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
