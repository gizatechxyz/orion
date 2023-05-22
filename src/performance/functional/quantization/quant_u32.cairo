use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use orion::operators::tensor::core::{Tensor, TensorTrait};
use orion::operators::tensor::implementations::impl_tensor_u32;
use orion::utils::u32_max;
use orion::utils::check_gas;

/// Symmetrically quantizes the input `data` value using the specified range.
///
/// # Arguments
/// * `min_val` - The minimum value of the input data range.
/// * `max_val` - The maximum value of the input data range.
/// * `data` - The u32 data value to be quantized.
///
/// # Panics
/// * Panics if the quantized value is out of the range of [-127, 127].
///
/// # Returns
/// * An u32 quantized value.
fn symetric_quant(min_val: u32, max_val: u32, data: u32) -> u32 {
    //  Define quantization range
    //  int8 range : [0;255] 
    let q_min_int = 0;
    let q_max_int = 255;

    let factor = 1000;
    let min_val = min_val * factor;
    let max_val = max_val * factor;

    //  Calculate the scale based on 8 bit symetric quantization
    //  scale = max(abs(data_range_max), abs(data_range_min)) * 2 / (quantization_range_max - quantization_range_min)

    let scale = (u32_max(min_val, max_val) * 2) / (q_max_int - q_min_int);

    //  Quantize data based on the scale
    let quantized_data = (data * factor) / scale;

    assert(quantized_data <= 255_u32, 'out of range');

    return quantized_data;
}

/// Cf: PerfomanceTrait::quantize_linear docstring
fn quantize_tensor(tensor: @Tensor::<u32>) -> Tensor::<u32> {
    let mut result_data = ArrayTrait::<u32>::new();

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
