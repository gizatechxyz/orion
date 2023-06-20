use core::debug::PrintTrait;
use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::numbers::fixed_point::core::{FixedTrait, FixedType};
use orion::numbers::fixed_point::math::math_16x16::max;
use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_fp;

use orion::numbers::fixed_point::implementations::impl_16x16;

use orion::utils::check_gas;

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
    let q_min_int = FixedTrait::new(1065353216, true); // -127
    let q_max_int = FixedTrait::new(1065353216, false); // 127

    //  Calculate the scale based on 8 bit symetric quantization
    //  scale = max(abs(data_range_max), abs(data_range_min)) * 2 / (quantization_range_max - quantization_range_min)

    let scale = (max(min_val.abs(), max_val.abs()) * FixedTrait::new_unscaled(2_u128, false))
        / (q_max_int - q_min_int);

    //  Quantize data based on the scale
    let quantized_data = data / scale;

    assert(quantized_data.mag < 1073741824, 'out of range');

    return quantized_data;
}

/// Cf: PerfomanceTrait::quantize_linear docstring
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

    return TensorTrait::new(*tensor.shape, result_data.span(), *tensor.extra);
}
