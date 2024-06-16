use orion::helpers::vec_f64_to_raw_data;
use orion::primgraph::{Dtype, Tensor};
use orion_test_utils::utils::prepare_test;
use std::f64::consts::PI;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        println!("Usage: {} <input_name> <output_name> <test_name>", args[0]);
        return;
    }
    match args[3].as_str() {
        "test_sin" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![Tensor {
                shape: vec![2, 2],
                raw_data: vec_f64_to_raw_data(&[0.0, PI / 2.0, PI, 3.0 * PI / 2.0]),
                dtype: Dtype::Double,
            }];

            prepare_test(input_name, output_name, input);
        }
        "test_add" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![2, 2],
                    raw_data: vec_f64_to_raw_data(&[1.0, 2.0, 3.0, 4.0]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![2, 2],
                    raw_data: vec_f64_to_raw_data(&[1.0, 2.0, 3.0, 4.0]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        "test_add_broadcast" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![2, 2],
                    raw_data: vec_f64_to_raw_data(&[1.0, 2.0, 3.0, 4.0]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![2],
                    raw_data: vec_f64_to_raw_data(&[1.0, 2.0]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        "test_add_broadcast_scalar" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![2, 2],
                    raw_data: vec_f64_to_raw_data(&[1.0, 2.0, 3.0, 4.0]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![1],
                    raw_data: vec_f64_to_raw_data(&[2.0]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        "test_mul" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![2, 2],
                    raw_data: vec_f64_to_raw_data(&[1.0, 2.0, 3.0, 4.0]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![2, 2],
                    raw_data: vec_f64_to_raw_data(&[2.0, 2.0, 4.0, 4.0]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        "test_mul_broadcast" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![2, 2],
                    raw_data: vec_f64_to_raw_data(&[1.0, 2.0, 3.0, 4.0]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![2],
                    raw_data: vec_f64_to_raw_data(&[1.0, 2.0]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        "test_mul_broadcast_scalar" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![2, 2],
                    raw_data: vec_f64_to_raw_data(&[1.0, 2.0, 3.0, 4.0]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![1],
                    raw_data: vec_f64_to_raw_data(&[2.0]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        "test_log" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![Tensor {
                shape: vec![2, 2],
                raw_data: vec_f64_to_raw_data(&[1.0, 2.0, 4.0, 8.0]),
                dtype: Dtype::Double,
            }];

            prepare_test(input_name, output_name, input);
        }
        "test_exp" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![Tensor {
                shape: vec![2, 2],
                raw_data: vec_f64_to_raw_data(&[0.0, 1.0, 2.0, 3.0]),
                dtype: Dtype::Double,
            }];

            prepare_test(input_name, output_name, input);
        }
        "test_sqrt" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![Tensor {
                shape: vec![2, 2],
                raw_data: vec_f64_to_raw_data(&[0.0, 1.0, 4.0, 9.0]),
                dtype: Dtype::Double,
            }];

            prepare_test(input_name, output_name, input);
        }
        "test_recip" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![Tensor {
                shape: vec![2, 2],
                raw_data: vec_f64_to_raw_data(&[0.5, 1.0, 4.0, 9.0]),
                dtype: Dtype::Double,
            }];

            prepare_test(input_name, output_name, input);
        }
        "test_mod" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![2, 1],
                    raw_data: vec_f64_to_raw_data(&[7.0, 10.0]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![1, 3],
                    raw_data: vec_f64_to_raw_data(&[3.0, 4.0, 5.0]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        "test_less" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![2, 1],
                    raw_data: vec_f64_to_raw_data(&[7.0, 10.0]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![1, 3],
                    raw_data: vec_f64_to_raw_data(&[3.0, 40.0, 5.0]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        "test_reducesum_keepdim" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![Tensor {
                shape: vec![3, 2, 2],
                raw_data: vec_f64_to_raw_data(&[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ]),
                dtype: Dtype::Double,
            }];

            prepare_test(input_name, output_name, input);
        }
        "test_reducesum_not_keepdim" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![Tensor {
                shape: vec![3, 2, 2],
                raw_data: vec_f64_to_raw_data(&[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ]),
                dtype: Dtype::Double,
            }];

            prepare_test(input_name, output_name, input);
        }
        "test_reducemax_keepdim" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![Tensor {
                shape: vec![3, 2, 2],
                raw_data: vec_f64_to_raw_data(&[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ]),
                dtype: Dtype::Double,
            }];

            prepare_test(input_name, output_name, input);
        }
        "test_reducemax_not_keepdim" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![Tensor {
                shape: vec![3, 2, 2],
                raw_data: vec_f64_to_raw_data(&[
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                ]),
                dtype: Dtype::Double,
            }];

            prepare_test(input_name, output_name, input);
        }
        "test_matmul_ax_a" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![4],
                    raw_data: vec_f64_to_raw_data(&[1.0, 2.0, 3.0, 4.0]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![4],
                    raw_data: vec_f64_to_raw_data(&[1.0, 2.0, 3.0, 4.0]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        "test_matmul_abx_b" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![3, 4],
                    raw_data: vec_f64_to_raw_data(&[
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                    ]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![4],
                    raw_data: vec_f64_to_raw_data(&[1.0, 2.0, 3.0, 4.0]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        "test_matmul_ax_ab" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![3],
                    raw_data: vec_f64_to_raw_data(&[1.0, 2.0, 3.0]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![3, 4],
                    raw_data: vec_f64_to_raw_data(&[
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                    ]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        "test_matmul_abcx_cd" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![3, 2, 2],
                    raw_data: vec_f64_to_raw_data(&[
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                    ]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![2, 6],
                    raw_data: vec_f64_to_raw_data(&[
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                    ]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        "test_matmul_abcx_acd" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![2, 3, 4],
                    raw_data: vec_f64_to_raw_data(&[
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 2.0,
                        3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                    ]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![2, 4, 3],
                    raw_data: vec_f64_to_raw_data(&[
                        4.0, 4.0, 4.0, 6.0, 6.0, 6.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 40.0, 40.0,
                        40.0, 60.0, 60.0, 60.0, 50.0, 50.0, 50.0, 100.0, 100.0, 100.0,
                    ]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        "test_matmul_abcdx_abde" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![2, 2, 3, 2],
                    raw_data: vec_f64_to_raw_data(&[
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 1.0, 2.0,
                        3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                    ]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![2, 2, 2, 3],
                    raw_data: vec_f64_to_raw_data(&[
                        4.0, 4.0, 4.0, 6.0, 6.0, 6.0, 5.0, 5.0, 5.0, 10.0, 10.0, 10.0, 40.0, 40.0,
                        40.0, 60.0, 60.0, 60.0, 50.0, 50.0, 50.0, 100.0, 100.0, 100.0,
                    ]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        "test_matmul_abcdex_abcef" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![2, 2, 2, 3, 2],
                    raw_data: vec_f64_to_raw_data(&[
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                        15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0,
                        27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0,
                        39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0,
                    ]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![2, 2, 2, 2, 3],
                    raw_data: vec_f64_to_raw_data(&[
                        24.0, 60.0, 69.0, 88.0, 3.0, 6.0, 21.0, 47.0, 30.0, 18.0, 53.0, 73.0, 9.0,
                        38.0, 77.0, 89.0, 61.0, 14.0, 39.0, 33.0, 36.0, 19.0, 50.0, 53.0, 10.0,
                        95.0, 44.0, 57.0, 66.0, 98.0, 95.0, 37.0, 93.0, 47.0, 49.0, 80.0, 5.0,
                        85.0, 58.0, 0.0, 1.0, 10.0, 86.0, 72.0, 3.0, 6.0, 82.0, 65.0,
                    ]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        "test_matmul" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![3, 3],
                    raw_data: vec_f64_to_raw_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![3, 3],
                    raw_data: vec_f64_to_raw_data(&[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        "test_matmul_identity_matrix" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![3, 3],
                    raw_data: vec_f64_to_raw_data(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![3, 3],
                    raw_data: vec_f64_to_raw_data(&[9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        "test_matmul_zero_matrix" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![3, 3],
                    raw_data: vec_f64_to_raw_data(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![3, 3],
                    raw_data: vec_f64_to_raw_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        "test_matmul_non_square_matrix" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![2, 3],
                    raw_data: vec_f64_to_raw_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![3, 2],
                    raw_data: vec_f64_to_raw_data(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        "test_matmul_initializer_A" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![Tensor {
                shape: vec![2, 3],
                raw_data: vec_f64_to_raw_data(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
                dtype: Dtype::Double,
            }];

            prepare_test(input_name, output_name, input);
        }
        "test_matmul_initializer_B" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![Tensor {
                shape: vec![3, 2],
                raw_data: vec_f64_to_raw_data(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
                dtype: Dtype::Double,
            }];

            prepare_test(input_name, output_name, input);
        }
        "test_matmul_multi_nodes" => {
            let input_name = &args[1];
            let output_name = &args[2];

            let input = vec![
                Tensor {
                    shape: vec![2, 3],
                    raw_data: vec_f64_to_raw_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                    dtype: Dtype::Double,
                },
                Tensor {
                    shape: vec![3, 2],
                    raw_data: vec_f64_to_raw_data(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
                    dtype: Dtype::Double,
                },
            ];

            prepare_test(input_name, output_name, input);
        }
        _ => {
            println!("Test Available for: {} : (test_1)", args[0]);
        }
    }
}
