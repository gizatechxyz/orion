use core::traits::Into;
use core::option::OptionTrait;
use array::SpanTrait;
use array::ArrayTrait;

use orion::numbers::signed_integer::i32::i32;
use orion::operators::tensor::core::{Tensor, TensorTrait, ExtraParams};
use orion::operators::tensor::implementations::impl_tensor_i32::{Tensor_i32, i32TensorAdd};
use orion::numbers::signed_integer::{integer_trait::IntegerTrait};
use orion::performance::core::PerfomanceTrait;

/// Cf: NNTrait::convint docstring
fn convint_i32(z: Tensor<i32>, weights: Tensor<i32>, bias: Tensor<i32>, kernel_size: usize, strides: usize) -> Tensor<i32> {
    assert(z.shape.len() >= 3, 'input tensor be at least 3D');

    let n_rows = *z.shape.at(0);
    let n_columns = *z.shape.at(1);
    let n_channels = *z.shape.at(2);
    let n_filters = *weights.shape.at(3);

    let i_max = (n_rows - kernel_size) / strides + 1;
    let j_max = (n_columns - kernel_size) / strides + 1;

    let mut i = 0_usize;

    let mut out_shape = ArrayTrait::new();
    out_shape.append(i_max);
    out_shape.append(j_max);
    out_shape.append(n_filters);
    let mut out_data = ArrayTrait::new();

    loop {
        if i == i_max {
            break ();
        }

        let mut j = 0_usize;
        loop {
            if j == j_max {
                break ();
            }

            let mut sum_shape = ArrayTrait::new();
            sum_shape.append(n_channels);
            sum_shape.append(n_filters);
            let mut sum_data = ArrayTrait::new();

            let mut k = 0_usize;
            loop {
                if k == n_channels {
                    break ();
                }
                let mut m = 0_usize;
                loop {
                    if m == n_filters {
                        break ();
                    }

                    let mut sum: i32 = IntegerTrait::new(0, false);

                    let mut x = 0_usize;
                    loop {
                        if x == kernel_size {
                            break ();
                        }

                        let mut y = 0_usize;
                        loop {
                            if y == kernel_size {
                                break ();
                            }

                            let mut input_indices = ArrayTrait::new();
                            input_indices.append(i * strides + x);
                            input_indices.append(j * strides + y);
                            input_indices.append(k);
                            let mut weight_indices = ArrayTrait::new();
                            weight_indices.append(x);
                            weight_indices.append(y);
                            weight_indices.append(k);
                            weight_indices.append(m);

                            sum += z.at(input_indices.span()) * weights.at(weight_indices.span());
                            y += 1;
                        };
                        x += 1;
                    };

                    sum_data.append(sum);

                    m += 1;
                };
                k += 1;
            };

            let sum_tensor = TensorTrait::new(sum_shape.span(), sum_data.span(), Option::None(()));

            let mut m = 0_usize;
            loop {
                if (m == n_filters) {
                    break ();
                }

                let mut k = 0_usize;
                let mut sum: i32 = IntegerTrait::new(0, false);

                loop {
                    if (k == n_channels) {
                        break ();
                    }

                    let mut sum_indices = ArrayTrait::new();
                    sum_indices.append(k);
                    sum_indices.append(m);

                    sum += sum_tensor.at(sum_indices.span());

                    k += 1;
                };

                out_data.append(sum);

                m += 1;
            };

            j += 1;
        };
        i += 1;
    };

    let output = TensorTrait::new(out_shape.span(), out_data.span(), Option::None(()));

    return output;
}
