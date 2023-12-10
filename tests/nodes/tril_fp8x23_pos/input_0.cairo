use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 192937984, sign: false });
    data.append(FP8x23 { mag: 159383552, sign: false });
    data.append(FP8x23 { mag: 142606336, sign: false });
    data.append(FP8x23 { mag: 444596224, sign: true });
    data.append(FP8x23 { mag: 796917760, sign: false });
    data.append(FP8x23 { mag: 587202560, sign: true });
    data.append(FP8x23 { mag: 889192448, sign: false });
    data.append(FP8x23 { mag: 872415232, sign: true });
    data.append(FP8x23 { mag: 964689920, sign: true });
    data.append(FP8x23 { mag: 486539264, sign: false });
    data.append(FP8x23 { mag: 67108864, sign: false });
    data.append(FP8x23 { mag: 796917760, sign: true });
    data.append(FP8x23 { mag: 310378496, sign: false });
    data.append(FP8x23 { mag: 947912704, sign: false });
    data.append(FP8x23 { mag: 998244352, sign: false });
    data.append(FP8x23 { mag: 176160768, sign: false });
    data.append(FP8x23 { mag: 226492416, sign: false });
    data.append(FP8x23 { mag: 343932928, sign: true });
    data.append(FP8x23 { mag: 1006632960, sign: false });
    data.append(FP8x23 { mag: 218103808, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
