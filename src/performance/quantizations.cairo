//  8 BIT LINEAR-QUANTIZATION
//  https://onnxruntime.ai/docs/performance/quantization.html#quantization-overview

use array::ArrayTrait;
use option::OptionTrait;
use onnx_cairo::operators::math::signed_integer::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32;
use onnx_cairo::operators::math::vector::find_min;
use onnx_cairo::operators::math::vector::find_max;

fn symetric_quant(min_val: i32, max_val: i32, data: i32) -> i32 {
    match gas::withdraw_gas_all(get_builtin_costs()) {
        Option::Some(x) => {},
        Option::None(x) => {
            let mut data = ArrayTrait::new();
            data.append('Out of gas');
            panic(data);
        },
    }

    //  Define quantization range
    //  int8 range : [-127;127] 
    let q_min_int = IntegerTrait::new(127_u32, true);
    let q_max_int = IntegerTrait::new(127_u32, false);

    let factor = IntegerTrait::new(1000_u32, false);
    let min_val = min_val * factor;
    let max_val = max_val * factor;

    //  Calculate the scale based on 8 bit symetric quantization
    //  scale = max(abs(data_range_max), abs(data_range_min)) * 2 / (quantization_range_max - quantization_range_min)

    let scale = ((min_val.abs()).max(max_val.abs()) * IntegerTrait::new(2_u32, false))
        / (q_max_int - q_min_int);

    //  Quantize data based on the scale
    let quantized_data = (data * factor) / scale;

    assert(quantized_data.mag <= 127_u32, 'out of range');

    return quantized_data;
}

fn quant_vec(vec: @Array::<i32>) -> Array::<i32> {
    match gas::withdraw_gas_all(get_builtin_costs()) {
        Option::Some(x) => {},
        Option::None(x) => {
            let mut data = ArrayTrait::new();
            data.append('Out of gas');
            panic(data);
        },
    }

    let mut result = ArrayTrait::<i32>::new();

    let mut min_val = find_min(vec);
    let mut max_val = find_max(vec);

    __quant_vec(ref min_val, ref max_val, vec, ref result, 0_usize);

    return result;
}

fn __quant_vec(
    ref min_val: i32, ref max_val: i32, vec: @Array::<i32>, ref result: Array::<i32>, n: usize
) {
    match gas::withdraw_gas_all(get_builtin_costs()) {
        Option::Some(x) => {},
        Option::None(x) => {
            let mut data = ArrayTrait::new();
            data.append('Out of gas');
            panic(data);
        },
    }

    // --- End of the recursion ---
    if n == vec.len() {
        return ();
    }

    // --- Quantize data and append to the result vector --- 
    let quantized = symetric_quant(min_val, max_val, *vec.at(n));
    result.append(quantized);

    // --- The process is repeated for the remaining elemets in the array --- 
    __quant_vec(ref min_val, ref max_val, vec, ref result, n + 1_usize)
}
