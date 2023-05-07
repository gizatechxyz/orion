use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::numbers::fixed_point::types::{Fixed, FixedType};
use onnx_cairo::numbers::fixed_point::core::max;
use onnx_cairo::operators::tensor::core::{Tensor, TensorTrait};
use onnx_cairo::operators::tensor::tensor_fp;
use onnx_cairo::utils::check_gas;

/// Symmetrically quantizes the input `data` value using the specified range.
///
/// # Arguments
/// * `min_val` - The minimum value of the input data range.
/// * `max_val` - The maximum value of the input data range.
/// * `data` - The FixedType data value to be quantized.
///
/// # Panics
/// * Panics if the quantized value is out of the range of [-127, 127].
///
/// # Returns
/// * An FixedType quantized value.
fn symetric_quant(min_val: FixedType, max_val: FixedType, data: FixedType) -> FixedType {
    //  Define quantization range
    //  int8 range : [-127;127] 
    let q_min_int = Fixed::new_unscaled(127_u128, true);
    let q_max_int = Fixed::new_unscaled(127_u128, false);

    let factor = Fixed::new_unscaled(1000_u128, false);
    let min_val = min_val * factor;
    let max_val = max_val * factor;

    //  Calculate the scale based on 8 bit symetric quantization
    //  scale = max(abs(data_range_max), abs(data_range_min)) * 2 / (quantization_range_max - quantization_range_min)

    let scale = (max(min_val.abs(), max_val.abs()) * Fixed::new_unscaled(2_u128, false))
        / (q_max_int - q_min_int);

    //  Quantize data based on the scale
    let quantized_data = (data * factor) / scale;

    assert(quantized_data.mag <= 127_u128, 'out of range');

    return quantized_data;
}

/// Quantizes an FixedType tensor using symmetric quantization.
///
/// # Arguments
/// * `tensor` - A reference to an FixedType tensor to be quantized.
///
/// # Panics
/// * Panics if gas limit is exceeded during execution.
///
/// # Returns
/// * A new FixedType tensor with the same shape as the input tensor, containing the quantized values.
fn quantize_tensor(tensor: @Tensor::<FixedType>) -> Tensor::<FixedType> {
    let mut result_data = ArrayTrait::<FixedType>::new();

    let min_val = tensor.min();
    let max_val = tensor.max();

    let mut data = *tensor.data;

    loop {
        check_gas();

        let quantized = symetric_quant(min_val, max_val, *data.pop_front().unwrap());
        result_data.append(quantized);

        if data.len() == 0 {
            break ();
        };
    };

    return TensorTrait::new(*tensor.shape, result_data.span());
}
