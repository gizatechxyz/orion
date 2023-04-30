use array::ArrayTrait;
use array::SpanTrait;
use option::OptionTrait;
use onnx_cairo::operators::tensor::core::Tensor;
use onnx_cairo::operators::tensor::core::TensorTrait;
use onnx_cairo::operators::tensor::tensor_u32;
use onnx_cairo::utils::u32_max;
use onnx_cairo::utils::check_gas;

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
