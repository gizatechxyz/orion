use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP16x16Tensor;
use orion::numbers::FixedTrait;
use orion::numbers::FP16x16;

fn input_0() -> Tensor<FP16x16> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(5);

    let mut data = ArrayTrait::new();
    data.append(FP16x16 { mag: 25149, sign: false });
    data.append(FP16x16 { mag: 57333, sign: false });
    data.append(FP16x16 { mag: 4965, sign: false });
    data.append(FP16x16 { mag: 43218, sign: false });
    data.append(FP16x16 { mag: 49951, sign: false });
    data.append(FP16x16 { mag: 61057, sign: false });
    data.append(FP16x16 { mag: 50263, sign: false });
    data.append(FP16x16 { mag: 29479, sign: false });
    data.append(FP16x16 { mag: 3849, sign: false });
    data.append(FP16x16 { mag: 38336, sign: false });
    data.append(FP16x16 { mag: 27897, sign: false });
    data.append(FP16x16 { mag: 9815, sign: false });
    data.append(FP16x16 { mag: 10500, sign: false });
    data.append(FP16x16 { mag: 46201, sign: false });
    data.append(FP16x16 { mag: 51565, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
