use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(1);
    shape.append(4);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 2107888, sign: true });
    data.append(FP8x23 { mag: 18609267, sign: false });
    data.append(FP8x23 { mag: 21110896, sign: false });
    data.append(FP8x23 { mag: 20658169, sign: true });
    data.append(FP8x23 { mag: 15019497, sign: false });
    data.append(FP8x23 { mag: 18600854, sign: false });
    data.append(FP8x23 { mag: 17219045, sign: false });
    data.append(FP8x23 { mag: 5826906, sign: false });
    data.append(FP8x23 { mag: 1835376, sign: false });
    data.append(FP8x23 { mag: 3485937, sign: false });
    data.append(FP8x23 { mag: 23249935, sign: true });
    data.append(FP8x23 { mag: 428809, sign: false });
    data.append(FP8x23 { mag: 20996700, sign: false });
    data.append(FP8x23 { mag: 7565588, sign: true });
    data.append(FP8x23 { mag: 15581476, sign: true });
    data.append(FP8x23 { mag: 7136954, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
