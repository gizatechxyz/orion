use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::FixedImpl;
use orion::operators::tensor::implementations::tensor_i8_fp16x16::Tensor_i8_fp16x16;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};

fn output_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 8, sign: false });
    data.append(i8 { mag: 117, sign: false });
    data.append(i8 { mag: 37, sign: true });
    data.append(i8 { mag: 57, sign: true });
    data.append(i8 { mag: 30, sign: true });
    data.append(i8 { mag: 92, sign: false });
    data.append(i8 { mag: 124, sign: false });
    data.append(i8 { mag: 28, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}