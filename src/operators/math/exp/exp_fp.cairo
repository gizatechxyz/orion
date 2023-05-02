use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use traits::Into;

use onnx_cairo::operators::math::fixed_point::types::Fixed;
use onnx_cairo::operators::math::fixed_point::types::FixedType;
use onnx_cairo::operators::tensor::helpers::check_shape;
use onnx_cairo::operators::tensor::helpers::check_compatibility;
use onnx_cairo::operators::tensor::helpers::broadcast_index_mapping;
use onnx_cairo::operators::tensor::core::Tensor;
use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::operators::tensor::core::ravel_index;
use onnx_cairo::operators::tensor::core::unravel_index;
use onnx_cairo::operators::tensor::tensor_fp;
use onnx_cairo::utils::check_gas;


/// Calculates the exponential function (e^x) for each element in a tensor of FixedType values.
///
/// # Arguments
///
/// * `self` - A tensor of FixedType values representing the input tensor.
///
/// # Panics
///
/// * If gas limit is reached during computation.
///
/// # Returns
///
/// * A tensor of fixed point numbers representing the result 
fn exp(self: @Tensor<FixedType>) -> Tensor<FixedType> {
    let mut result = ArrayTrait::new();
    let mut data = *self.data;

    loop {
        check_gas();

        let ele = *data.pop_front().unwrap();
        result.append(Fixed::exp(ele));

        if (data.len() == 0) {
            break ();
        };
    };

    return TensorTrait::<FixedType>::new(*self.shape, result.span());
}
