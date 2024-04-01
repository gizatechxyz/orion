use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(3);
    shape.append(2);
    shape.append(1);
    shape.append(1);
    shape.append(1);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 7649074, sign: true });
    data.append(FP8x23 { mag: 15131138, sign: true });
    data.append(FP8x23 { mag: 6851097, sign: false });
    data.append(FP8x23 { mag: 3652233, sign: true });
    data.append(FP8x23 { mag: 6761260, sign: false });
    data.append(FP8x23 { mag: 11060859, sign: true });
    data.append(FP8x23 { mag: 4130910, sign: false });
    data.append(FP8x23 { mag: 5583802, sign: true });
    data.append(FP8x23 { mag: 7717068, sign: true });
    data.append(FP8x23 { mag: 10798484, sign: true });
    data.append(FP8x23 { mag: 14302407, sign: false });
    data.append(FP8x23 { mag: 8104245, sign: true });
    data.append(FP8x23 { mag: 10993385, sign: false });
    data.append(FP8x23 { mag: 9869012, sign: true });
    data.append(FP8x23 { mag: 7247863, sign: true });
    data.append(FP8x23 { mag: 2273946, sign: false });
    data.append(FP8x23 { mag: 7044627, sign: true });
    data.append(FP8x23 { mag: 15165248, sign: false });
    data.append(FP8x23 { mag: 5982575, sign: false });
    data.append(FP8x23 { mag: 7138613, sign: true });
    data.append(FP8x23 { mag: 11200383, sign: false });
    data.append(FP8x23 { mag: 25522932, sign: false });
    data.append(FP8x23 { mag: 3612890, sign: false });
    data.append(FP8x23 { mag: 11607214, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
