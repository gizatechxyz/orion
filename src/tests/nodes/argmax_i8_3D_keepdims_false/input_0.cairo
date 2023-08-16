use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::FixedImpl;
use orion::operators::tensor::implementations::impl_tensor_i8::Tensor_i8;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i8::i8};

fn input_0() -> Tensor<i8> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i8 { mag: 118, sign: false });
    data.append(i8 { mag: 72, sign: false });
    data.append(i8 { mag: 115, sign: false });
    data.append(i8 { mag: 60, sign: false });
    data.append(i8 { mag: 48, sign: false });
    data.append(i8 { mag: 51, sign: false });
    data.append(i8 { mag: 112, sign: true });
    data.append(i8 { mag: 72, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}