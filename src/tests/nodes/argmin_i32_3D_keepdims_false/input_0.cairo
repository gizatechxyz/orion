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
    data.append(i32 { mag: 76, sign: true });
    data.append(i32 { mag: 55, sign: false });
    data.append(i32 { mag: 11, sign: true });
    data.append(i32 { mag: 64, sign: false });
    data.append(i32 { mag: 56, sign: true });
    data.append(i32 { mag: 37, sign: false });
    data.append(i32 { mag: 30, sign: true });
    data.append(i32 { mag: 75, sign: true });

    let extra = ExtraParams { fixed_point: Option::Some(FixedImpl::FP16x16) };
    TensorTrait::new(shape.span(), data.span(), Option::Some(extra))
}