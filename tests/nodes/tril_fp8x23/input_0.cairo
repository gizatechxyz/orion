use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::{FP8x23Tensor, FP8x23TensorAdd};
use orion::numbers::{FixedTrait, FP8x23};

fn input_0() -> Tensor<FP8x23> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(4);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP8x23 { mag: 838860800, sign: false });
    data.append(FP8x23 { mag: 218103808, sign: true });
    data.append(FP8x23 { mag: 1065353216, sign: true });
    data.append(FP8x23 { mag: 67108864, sign: true });
    data.append(FP8x23 { mag: 905969664, sign: true });
    data.append(FP8x23 { mag: 637534208, sign: true });
    data.append(FP8x23 { mag: 352321536, sign: false });
    data.append(FP8x23 { mag: 973078528, sign: true });
    data.append(FP8x23 { mag: 209715200, sign: false });
    data.append(FP8x23 { mag: 218103808, sign: true });
    data.append(FP8x23 { mag: 192937984, sign: false });
    data.append(FP8x23 { mag: 813694976, sign: false });
    data.append(FP8x23 { mag: 184549376, sign: true });
    data.append(FP8x23 { mag: 184549376, sign: true });
    data.append(FP8x23 { mag: 671088640, sign: true });
    data.append(FP8x23 { mag: 176160768, sign: false });
    data.append(FP8x23 { mag: 880803840, sign: true });
    data.append(FP8x23 { mag: 637534208, sign: false });
    data.append(FP8x23 { mag: 360710144, sign: false });
    data.append(FP8x23 { mag: 973078528, sign: true });
    TensorTrait::new(shape.span(), data.span())
}
