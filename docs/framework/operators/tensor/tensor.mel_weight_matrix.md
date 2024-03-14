# TensorTrait::mel_weight_matrix

```rust
        fn mel_weight_matrix(num_mel_bins: usize, dft_length: usize, sample_rate: usize, lower_edge_hertz: T, upper_edge_hertz: T) -> Tensor<T>;
```

Generate a MelWeightMatrix that can be used to re-weight a Tensor containing a linearly sampled frequency spectra (from DFT or STFT) into num_mel_bins frequency information based on the [lower_edge_hertz, upper_edge_hertz] range on the mel scale. 
This function defines the mel scale in terms of a frequency in hertz according to the following formula:
```
mel(f) = 2595 * log10(1 + f/700)
```
In the returned matrix, all the triangles (filterbanks) have a peak value of 1.0.
The returned MelWeightMatrix can be used to right-multiply a spectrogram S of shape [frames, num_spectrogram_bins] of linear scale spectrum values (e.g. STFT magnitudes) to generate a “mel spectrogram” M of shape [frames, num_mel_bins].
## Args

* `num_mel_bins `(`usize`) - The number of bands in the mel spectrum.
* `dft_length `(`usize`) - The size of the original DFT. The size of the original DFT is used to infer the size of the onesided DFT, which is understood to be floor(dft_length/2) + 1, i.e. the spectrogram only contains the nonredundant DFT bins.
* `sample_rate `(`usize`) - Samples per second of the input signal used to create the spectrogram. Used to figure out the frequencies corresponding to each spectrogram bin, which dictates how they are mapped into the mel scale.
* `lower_edge_hertz `(T) - Lower bound on the frequencies to be included in the mel spectrum. This corresponds to the lower edge of the lowest triangular band.
* `upper_edge_hertz  `(T) - The desired top edge of the highest frequency band.

## Returns

* A `Tensor<T>` The Mel Weight Matrix. The output has the shape: [floor(dft_length/2) + 1][num_mel_bins].

## Examples

```rust
use orion::operators::tensor::{FP16x16Tensor, FP16x16TensorAdd};
use core::array::{ArrayTrait, SpanTrait};
use orion::operators::tensor::FP16x16TensorPartialEq;
use orion::utils::{assert_eq, assert_seq_eq};
use orion::operators::tensor::{TensorTrait, Tensor};
use orion::numbers::{FixedTrait, FP16x16};


fn example() -> Tensor<FP16x16> {
    return TensorTrait::mel_weight_matrix(8, 16, 8192, FP16x16 { mag: 0, sign: false }, FP16x16 { mag: 268435456, sign: false });
}
>>> [
        [65536, 65536, 0, 0, 0, 0, 0, 0],
        [0, 0, 65536, 65536, 0, 0, 0, 0],   
        [0, 0, 0, 0, 65536, 0, 0, 0],
        [0, 0, 0, 0, 0, 65536, 0, 0],
        [0, 0, 0, 0, 0, 0, 65536, 0],   
        [0, 0, 0, 0, 0, 0, 0, 65536],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ]
```
