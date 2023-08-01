use array::ArrayTrait;
use orion::operators::tensor::core::{TensorTrait, Tensor, ExtraParams};
use orion::numbers::fixed_point::core::FixedImpl;
use orion::operators::tensor::implementations::impl_tensor_i32::Tensor_i32;
use orion::numbers::signed_integer::{integer_trait::IntegerTrait, i32::i32};

fn input_0() -> Tensor<i32> {
    let mut shape = ArrayTrait::<usize>::new();
    shape.append(2);
    shape.append(2);
    shape.append(2);

    let mut data = ArrayTrait::new();
    data.append(i32 { mag: 26, sign: false });
    data.append(i32 { mag: 55, sign: false });
    data.append(i32 { mag: 57, sign: false });
    data.append(i32 { mag: 89, sign: true });
    data.append(i32 { mag: 73, sign: false });
    data.append(i32 { mag: 82, sign: false });
    data.append(i32 { mag: 92, sign: false });
    data.append(i32 { mag: 70, sign: false });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}