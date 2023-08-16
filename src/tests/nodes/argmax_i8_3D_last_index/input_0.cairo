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
    data.append(i8 { mag: 14, sign: false });
    data.append(i8 { mag: 43, sign: true });
    data.append(i8 { mag: 126, sign: true });
    data.append(i8 { mag: 49, sign: false });
    data.append(i8 { mag: 127, sign: true });
    data.append(i8 { mag: 88, sign: true });
    data.append(i8 { mag: 79, sign: false });
    data.append(i8 { mag: 61, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}