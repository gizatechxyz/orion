use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;

use onnx_cairo::operators::math::signed_integer::integer_trait::IntegerTrait;
use onnx_cairo::operators::math::signed_integer::i32::i32Impl;
use onnx_cairo::operators::math::signed_integer::i32::i32;
use onnx_cairo::operators::tensor::core::Tensor;
use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::operators::tensor::tensor_i32;
use onnx_cairo::utils::check_gas;

fn symetric_quant(min_val: i32, max_val: i32, data: i32) -> i32 {
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

fn quantize_tensor(tensor: @Tensor::<i32>) -> Tensor::<i32> {
    let mut result_data = ArrayTrait::<i32>::new();

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
