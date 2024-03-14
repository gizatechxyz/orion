import numpy as np
from nodegen.node import RunAll
from ..helpers import make_test, to_fp, Tensor, Dtype, FixedImpl, Trait, get_data_statement

def mel_weight_matrix(num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz) -> np.ndarray:  # type: ignore
    # num_mel_bins = np.int32(8)
    # dft_length = np.int32(16)
    # sample_rate = np.int32(8192)
    # lower_edge_hertz = np.float32(0)
    # upper_edge_hertz = np.float32(8192 / 2)
    num_spectrogram_bins = dft_length // 2 + 1
    frequency_bins = np.arange(0, num_mel_bins + 2)

    low_frequency_mel = 2595 * np.log10(1 + lower_edge_hertz / 700)
    high_frequency_mel = 2595 * np.log10(1 + upper_edge_hertz / 700)
    mel_step = (high_frequency_mel - low_frequency_mel) / frequency_bins.shape[0]

    frequency_bins = frequency_bins * mel_step + low_frequency_mel
    frequency_bins = 700 * (np.power(10, (frequency_bins / 2595)) - 1)
    frequency_bins = ((dft_length + 1) * frequency_bins) // sample_rate
    frequency_bins = frequency_bins.astype(int)

    output = np.zeros((num_spectrogram_bins, num_mel_bins))
    output.flags.writeable = True

    for i in range(num_mel_bins):
        lower_frequency_value = frequency_bins[i]  # left
        center_frequency_point = frequency_bins[i + 1]  # center
        higher_frequency_point = frequency_bins[i + 2]  # right
        low_to_center = center_frequency_point - lower_frequency_value
        if low_to_center == 0:
            output[center_frequency_point, i] = 1
        else:
            for j in range(lower_frequency_value, center_frequency_point + 1):
                output[j, i] = float(j - lower_frequency_value) / float(
                    low_to_center
                )
        center_to_high = higher_frequency_point - center_frequency_point
        if center_to_high > 0:
            for j in range(center_frequency_point, higher_frequency_point):
                output[j, i] = float(higher_frequency_point - j) / float(
                    center_to_high
                )
    return output


class Mel_weight_matrix(RunAll):
     
    # @staticmethod
    # # We test here with u32 implementation.
    # def u32():
    #     num_mel_bins = np.int32(8)
    #     dft_length = np.int32(16)
    #     sample_rate = np.int32(256)
    #     lower_edge_hertz = np.int32(0)
    #     upper_edge_hertz = np.int32(256 / 2)
    #     args_str = get_data_statement(np.array([lower_edge_hertz, upper_edge_hertz]), Dtype.U32)
    #     y = mel_weight_matrix(num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz)
    #     print(y)
        
    #     # Convert the floats values in `y` to fixed points with `to_fp` method:
    #     y = Tensor(Dtype.U32, y.shape, y.flatten())
        
    #     # Define the name of the generated folder. 
    #     name = "mel_weight_matrix_u32"
    #     # Invoke `make_test` method to generate corresponding Cairo tests:
    #     make_test(
    #         [], # List of input tensors.
    #         y, # The expected output result.
    #         f"TensorTrait::mel_weight_matrix({f'{num_mel_bins}, {dft_length}, {sample_rate}, '+', '.join(args_str)})", # The code signature.
    #         name # The name of the generated folder.
    #     )
     
    @staticmethod
    # We test here with fp16x16 implementation.
    def fp16x16():
        num_mel_bins = np.int32(8)
        dft_length = np.int32(16)
        sample_rate = np.int32(8192)
        lower_edge_hertz = np.float32(0)
        upper_edge_hertz = np.float32(8192 / 2)
        args_str = get_data_statement(to_fp(np.array([lower_edge_hertz, upper_edge_hertz]).flatten(), FixedImpl.FP16x16), Dtype.FP16x16)
        y = mel_weight_matrix(num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz)
        print(y)
        
        # Convert the floats values in `y` to fixed points with `to_fp` method:
        y = Tensor(Dtype.FP16x16, y.shape, to_fp(y.flatten(), FixedImpl.FP16x16))
        
        # Define the name of the generated folder. 
        name = "mel_weight_matrix_fp16x16"
        # Invoke `make_test` method to generate corresponding Cairo tests:
        make_test(
            [], # List of input tensors.
            y, # The expected output result.
            f"TensorTrait::mel_weight_matrix({f'{num_mel_bins}, {dft_length}, {sample_rate}, '+', '.join(args_str)})", # The code signature.
            name # The name of the generated folder.
        )