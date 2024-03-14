use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::NumberTrait;
use orion::operators::tensor::core::{Tensor, TensorTrait};

#[derive(Copy, Drop)]

fn mel_weight_matrix<
    T,
    MAG,
    impl TTensor: TensorTrait<T>,
    impl TNumber: NumberTrait<T, MAG>,
    impl TAdd: Add<T>,
    impl TSub: Sub<T>,
    impl TMul: Mul<T>,
    impl TDiv: Div<T>,
    impl TTensorAdd: Add<Tensor<T>>,
    impl TPartialOrd: PartialOrd<T>,
    impl TAddEq: AddEq<T>,
    impl TCopy: Copy<T>,
    impl TDrop: Drop<T>,
    impl TTryInto: TryInto<T, usize>,
    impl TInto: Into<usize, MAG>,
>(
    num_mel_bins: usize, 
    dft_length: usize, 
    sample_rate: usize,
    lower_edge_hertz: T,
    upper_edge_hertz: T,
) -> Tensor<T> {
    let num_spectrogram_bins: usize = dft_length / 2 + 1;
    let range_len = num_mel_bins + 2;
    let range_len_T = NumberTrait::new_unscaled(range_len.into(), false);
    let mut frequency_bins: Tensor<T> = TensorTrait::range(NumberTrait::zero(), range_len_T, NumberTrait::one());
    let mut data: Array<T> = array![];

    // mel = 2595 * log10(1+f/700) 
    let mel_magic: usize = 2595;
    let corner_frequency: usize = 700;
    let power: usize = 10;
    let mel_magic_T = NumberTrait::new_unscaled(mel_magic.into(), false);
    let corner_frequency_T = NumberTrait::new_unscaled(corner_frequency.into(), false);
    let power_T = NumberTrait::new_unscaled(power.into(), false);

    let low_frequency_mel: T = mel_magic_T * NumberTrait::log10(NumberTrait::one() + lower_edge_hertz / corner_frequency_T);
    let high_frequency_mel: T = mel_magic_T * NumberTrait::log10(NumberTrait::one() + upper_edge_hertz / corner_frequency_T);
    let mel_step: T = (high_frequency_mel - low_frequency_mel) / NumberTrait::new_unscaled((*frequency_bins.shape[0]).into(), false);


    let mut frequency_bins_temp: Array<usize> = array![];
    let mut ones: Array<(usize, usize)> = array![];
    loop {
        match (frequency_bins.data).pop_front() {
            Option::Some(item) => {
                let mut frequency_bin: T = *item * mel_step + low_frequency_mel;
                frequency_bin = corner_frequency_T * (NumberTrait::pow(power_T, (frequency_bin / mel_magic_T)) - NumberTrait::one());
                frequency_bin = (NumberTrait::new_unscaled((dft_length + 1).into(), false) * frequency_bin) / NumberTrait::new_unscaled(sample_rate.into(), false);
                let frequency_bin_int_type: u32 = NumberTrait::floor(frequency_bin).try_into().unwrap();
                frequency_bins_temp.append(frequency_bin_int_type); 
            },
            Option::None => { break; }
        };
    };
    let mut index = 0;
    while index < num_mel_bins{
        let lower_frequency_value = *frequency_bins_temp[index];  // left
        let center_frequency_point = *frequency_bins_temp[index + 1];  // center
        let higher_frequency_point = *frequency_bins_temp[index + 2];  // right
        let low_to_center = center_frequency_point - lower_frequency_value;

        if low_to_center == 0{
            ones.append((center_frequency_point, index));
        } else {
            let mut i = lower_frequency_value;
            while center_frequency_point + 1 > i {
                if (i-lower_frequency_value) / low_to_center == 1 {
                    ones.append((i, index));
                }
                i += 1;
            };
        }
        let center_to_high = higher_frequency_point - center_frequency_point;
        if center_to_high > 0 {
            let mut i = center_frequency_point;
            while higher_frequency_point > i {
                if (higher_frequency_point-i) / center_to_high == 1 {
                    ones.append((i, index));
                }
                i += 1;
            };
        }
        index += 1;

    };
    // Fall in data
    loop {
        match (ones).pop_front() {
            // num_mel_bins * X + Y >= data_len
            Option::Some((x, y)) => {
                
                if num_mel_bins * x + y > data.len() {
                    let mut distance = num_mel_bins * x + y - data.len();
                    while distance > 0 {
                        data.append(NumberTrait::zero());
                        distance -= 1;
                    };
                }
                
                if num_mel_bins * x + y >= data.len(){
                    data.append(NumberTrait::one());
                }

                 
            },
            Option::None => { 
                // Complement
                let output_len = num_spectrogram_bins*num_mel_bins;
                while output_len > data.len() {
                    data.append(NumberTrait::zero());
                };
                break; 
            }
        };
    };

    TensorTrait::<T>::new(array![num_spectrogram_bins, num_mel_bins].span(), data.span())

}


