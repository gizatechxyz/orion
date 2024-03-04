use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::operators::tensor::FP32x32Tensor;
use orion::numbers::{FixedTrait, FP32x32};

fn output_0() -> Tensor<FP32x32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(3);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(FP32x32 { mag: 9935383294, sign: false });
    data.append(FP32x32 { mag: 18525317886, sign: false });
    data.append(FP32x32 { mag: 27115252478, sign: false });
    data.append(FP32x32 { mag: 35705187070, sign: false });
    data.append(FP32x32 { mag: 44295121662, sign: false });
    data.append(FP32x32 { mag: 52885056254, sign: false });
    TensorTrait::new(shape.span(), data.span())
}
