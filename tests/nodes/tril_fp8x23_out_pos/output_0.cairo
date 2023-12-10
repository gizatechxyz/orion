use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP8x23Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP8x23;

fn output_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 872415232, sign: false });
    data.append(FP8x23 { mag: 251658240, sign: true });
    data.append(FP8x23 { mag: 192937984, sign: false });
    data.append(FP8x23 { mag: 184549376, sign: true });
    data.append(FP8x23 { mag: 931135488, sign: false });
    data.append(FP8x23 { mag: 1015021568, sign: false });
    data.append(FP8x23 { mag: 310378496, sign: true });
    data.append(FP8x23 { mag: 67108864, sign: false });
    data.append(FP8x23 { mag: 1006632960, sign: true });
    data.append(FP8x23 { mag: 822083584, sign: true });
    data.append(FP8x23 { mag: 973078528, sign: false });
    data.append(FP8x23 { mag: 662700032, sign: true });
    data.append(FP8x23 { mag: 704643072, sign: false });
    data.append(FP8x23 { mag: 260046848, sign: false });
    data.append(FP8x23 { mag: 478150656, sign: false });
    data.append(FP8x23 { mag: 562036736, sign: true });
    data.append(FP8x23 { mag: 587202560, sign: true });
    data.append(FP8x23 { mag: 905969664, sign: true });
    data.append(FP8x23 { mag: 100663296, sign: true });
    data.append(FP8x23 { mag: 864026624, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
