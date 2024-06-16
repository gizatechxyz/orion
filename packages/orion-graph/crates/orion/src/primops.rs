use crate::{
    helpers::{
        broadcast_index_mapping, len_from_shape, max_along_axis, sum_along_axis, unravel_index,
        vec_bool_to_raw_data, vec_f64_to_raw_data, vec_raw_data_to_f64, vec_raw_data_to_i32,
    },
    primgraph::{Dtype, Tensor},
};
pub trait PrimopsTrait {
    fn log2(&self) -> Tensor;
    fn exp2(&self) -> Tensor;
    fn sin(&self) -> Tensor;
    fn sqrt(&self) -> Tensor;
    fn recip(&self) -> Tensor;
    fn add(&self, other: Tensor, output_shape: Vec<usize>) -> Tensor;
    fn mul(&self, other: Tensor, output_shape: Vec<usize>) -> Tensor;
    fn modulo(&self, other: Tensor, output_shape: Vec<usize>) -> Tensor;
    fn lessthan(&self, other: Tensor, output_shape: Vec<usize>) -> Tensor;
    fn sum_reduce(
        &self,
        axes: Option<Tensor>,
        keepdims: Option<bool>,
        noop_with_empty_axes: Option<bool>,
    ) -> Tensor;
    fn max_reduce(
        &self,
        axes: Option<Tensor>,
        keepdims: Option<bool>,
        noop_with_empty_axes: Option<bool>,
    ) -> Tensor;
}

impl PrimopsTrait for Tensor {
    fn log2(&self) -> Tensor {
        let data_f64 = vec_raw_data_to_f64(&self.raw_data);
        let data_f64: Vec<f64> = data_f64.iter().map(|&val| val.log2()).collect();

        Tensor {
            shape: self.shape.clone(),
            raw_data: vec_f64_to_raw_data(&data_f64),
            dtype: Dtype::Double,
        }
    }

    fn exp2(&self) -> Tensor {
        let data_f64 = vec_raw_data_to_f64(&self.raw_data);
        let data_f64: Vec<f64> = data_f64.iter().map(|&val| val.exp2()).collect();

        Tensor {
            shape: self.shape.clone(),
            raw_data: vec_f64_to_raw_data(&data_f64),
            dtype: Dtype::Double,
        }
    }

    fn sin(&self) -> Tensor {
        let data_f64 = vec_raw_data_to_f64(&self.raw_data);
        let data_f64: Vec<f64> = data_f64.iter().map(|&val| val.sin()).collect();

        Tensor {
            shape: self.shape.clone(),
            raw_data: vec_f64_to_raw_data(&data_f64),
            dtype: Dtype::Double,
        }
    }

    fn sqrt(&self) -> Tensor {
        let data_f64 = vec_raw_data_to_f64(&self.raw_data);
        let data_f64: Vec<f64> = data_f64.iter().map(|&val| val.sqrt()).collect();

        Tensor {
            shape: self.shape.clone(),
            raw_data: vec_f64_to_raw_data(&data_f64),
            dtype: Dtype::Double,
        }
    }

    fn recip(&self) -> Tensor {
        let data_f64 = vec_raw_data_to_f64(&self.raw_data);
        let data_f64: Vec<f64> = data_f64.iter().map(|&val| val.recip()).collect();

        Tensor {
            shape: self.shape.clone(),
            raw_data: vec_f64_to_raw_data(&data_f64),
            dtype: Dtype::Double,
        }
    }

    fn add(&self, other: Tensor, output_shape: Vec<usize>) -> Tensor {
        let mut data_f64 = vec![];

        let self_data_f64 = vec_raw_data_to_f64(&self.raw_data);
        let other_data_f64 = vec_raw_data_to_f64(&other.raw_data);

        let num_elements = len_from_shape(&output_shape);

        let mut n = 0;
        while n != num_elements {
            let indices_broadcasted = unravel_index(n, &output_shape);

            let indices_self =
                broadcast_index_mapping(self.shape.clone(), indices_broadcasted.clone());
            let indices_other = broadcast_index_mapping(other.shape.clone(), indices_broadcasted);
            data_f64.push(
                self_data_f64.get(indices_self).unwrap()
                    + other_data_f64.get(indices_other).unwrap(),
            );

            n += 1;
        }

        Tensor {
            shape: output_shape,
            raw_data: vec_f64_to_raw_data(&data_f64),
            dtype: Dtype::Double,
        }
    }

    fn mul(&self, other: Tensor, output_shape: Vec<usize>) -> Tensor {
        let mut data_f64 = vec![];

        let self_data_f64 = vec_raw_data_to_f64(&self.raw_data);
        let other_data_f64 = vec_raw_data_to_f64(&other.raw_data);

        let num_elements = len_from_shape(&output_shape);

        let mut n = 0;
        while n != num_elements {
            let indices_broadcasted = unravel_index(n, &output_shape);

            let indices_self =
                broadcast_index_mapping(self.shape.clone(), indices_broadcasted.clone());
            let indices_other = broadcast_index_mapping(other.shape.clone(), indices_broadcasted);
            data_f64.push(
                self_data_f64.get(indices_self).unwrap()
                    * other_data_f64.get(indices_other).unwrap(),
            );

            n += 1;
        }

        Tensor {
            shape: output_shape,
            raw_data: vec_f64_to_raw_data(&data_f64),
            dtype: Dtype::Double,
        }
    }

    fn modulo(&self, other: Tensor, output_shape: Vec<usize>) -> Tensor {
        let mut data_f64 = vec![];

        let self_data_f64 = vec_raw_data_to_f64(&self.raw_data);
        let other_data_f64 = vec_raw_data_to_f64(&other.raw_data);

        let num_elements = len_from_shape(&output_shape);

        let mut n = 0;
        while n != num_elements {
            let indices_broadcasted = unravel_index(n, &output_shape);

            let indices_self =
                broadcast_index_mapping(self.shape.clone(), indices_broadcasted.clone());
            let indices_other = broadcast_index_mapping(other.shape.clone(), indices_broadcasted);
            data_f64.push(
                self_data_f64.get(indices_self).unwrap()
                    % other_data_f64.get(indices_other).unwrap(),
            );

            n += 1;
        }

        Tensor {
            shape: output_shape,
            raw_data: vec_f64_to_raw_data(&data_f64),
            dtype: Dtype::Double,
        }
    }

    fn lessthan(&self, other: Tensor, output_shape: Vec<usize>) -> Tensor {
        let mut data_f64 = vec![];

        let self_data_f64 = vec_raw_data_to_f64(&self.raw_data);
        let other_data_f64 = vec_raw_data_to_f64(&other.raw_data);

        let num_elements = len_from_shape(&output_shape);

        let mut n = 0;
        while n != num_elements {
            let indices_broadcasted = unravel_index(n, &output_shape);

            let indices_self =
                broadcast_index_mapping(self.shape.clone(), indices_broadcasted.clone());
            let indices_other = broadcast_index_mapping(other.shape.clone(), indices_broadcasted);
            data_f64.push(
                self_data_f64.get(indices_self).unwrap()
                    < other_data_f64.get(indices_other).unwrap(),
            );

            n += 1;
        }

        Tensor {
            shape: output_shape,
            raw_data: vec_bool_to_raw_data(&data_f64),
            dtype: Dtype::Bool,
        }
    }

    fn sum_reduce(
        &self,
        axes: Option<Tensor>,
        keepdims: Option<bool>,
        _noop: Option<bool>,
    ) -> Tensor {
        let self_data_f64 = vec_raw_data_to_f64(&self.raw_data);
        let noop = _noop.unwrap_or(false);

        let axes: Vec<usize> = match axes {
            Some(axes) => {
                let axes = vec_raw_data_to_i32(&axes.raw_data);
                axes.iter()
                    .map(|&axis| {
                        if axis < 0 {
                            (self.shape.len() as i32 + axis) as usize
                        } else {
                            axis as usize
                        }
                    })
                    .collect()
            }
            None => {
                if noop {
                    return (*self).clone();
                }
                (0..self.shape.len()).collect()
            }
        };

        let keepdims = keepdims.unwrap_or(false);

        let mut result_data_f64 = self_data_f64.clone();
        let mut result_shape = self.shape.clone();

        for &axis in axes.iter().rev() {
            result_data_f64 = sum_along_axis(&result_data_f64, &result_shape, axis);
            if keepdims {
                result_shape[axis] = 1;
            } else {
                result_shape.remove(axis);
            }
        }

        Tensor {
            shape: result_shape,
            raw_data: vec_f64_to_raw_data(&result_data_f64),
            dtype: Dtype::Double,
        }
    }

    fn max_reduce(
        &self,
        axes: Option<Tensor>,
        keepdims: Option<bool>,
        _noop: Option<bool>,
    ) -> Tensor {
        let self_data_f64 = vec_raw_data_to_f64(&self.raw_data);
        let noop = _noop.unwrap_or(false);

        let axes: Vec<usize> = match axes {
            Some(axes) => {
                let axes = vec_raw_data_to_i32(&axes.raw_data);
                axes.iter()
                    .map(|&axis| {
                        if axis < 0 {
                            (self.shape.len() as i32 + axis) as usize
                        } else {
                            axis as usize
                        }
                    })
                    .collect()
            }
            None => {
                if noop {
                    return (*self).clone();
                }
                (0..self.shape.len()).collect()
            }
        };

        let keepdims = keepdims.unwrap_or(false);

        let mut result_data_f64 = self_data_f64.clone();
        let mut result_shape = self.shape.clone();

        for &axis in axes.iter().rev() {
            result_data_f64 = max_along_axis(&result_data_f64, &result_shape, axis);
            if keepdims {
                result_shape[axis] = 1;
            } else {
                result_shape.remove(axis);
            }
        }

        Tensor {
            shape: result_shape,
            raw_data: vec_f64_to_raw_data(&result_data_f64),
            dtype: Dtype::Double,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::vec_i32_into_raw_data;
    use std::f64::consts::PI;

    #[test]
    fn test_log2() {
        let tensor = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![1.0, 2.0, 4.0, 8.0]),
            dtype: Dtype::Double,
        };

        let result = tensor.log2();

        let expected_tensor = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![0.0, 1.0, 2.0, 3.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn test_exp2() {
        let tensor = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![0.0, 1.0, 2.0, 3.0]),
            dtype: Dtype::Double,
        };

        let result = tensor.exp2();

        let expected_tensor = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![1.0, 2.0, 4.0, 8.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn test_sin() {
        let tensor = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0]),
            dtype: Dtype::Double,
        };

        let result = tensor.sin();

        let expected_tensor = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![0.0, 1.0, 0.0, -1.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert!(vec_raw_data_to_f64(&result.raw_data)
            .iter()
            .zip(vec_raw_data_to_f64(&expected_tensor.raw_data).iter())
            .all(|(a, b)| (a - b).abs() < 1e-10));
    }

    #[test]
    fn test_sqrt() {
        let tensor = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![0.0, 1.0, 4.0, 9.0]),
            dtype: Dtype::Double,
        };

        let result = tensor.sqrt();

        let expected_tensor = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![0.0, 1.0, 2.0, 3.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn test_recip() {
        let tensor = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![1.0, 2.0, 4.0, 0.5]),
            dtype: Dtype::Double,
        };

        let result = tensor.recip();

        let expected_tensor = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![1.0, 0.5, 0.25, 2.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn test_add_same_shape() {
        let tensor1 = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![1.0, 2.0, 3.0, 4.0]),
            dtype: Dtype::Double,
        };

        let tensor2 = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![5.0, 6.0, 7.0, 8.0]),
            dtype: Dtype::Double,
        };

        let result = tensor1.add(tensor2, vec![2, 2]);

        let expected_tensor = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![6.0, 8.0, 10.0, 12.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn test_add_broadcasting() {
        let tensor1 = Tensor {
            shape: vec![2, 1],
            raw_data: vec_f64_to_raw_data(&vec![1.0, 2.0]),
            dtype: Dtype::Double,
        };

        let tensor2 = Tensor {
            shape: vec![1, 3],
            raw_data: vec_f64_to_raw_data(&vec![3.0, 4.0, 5.0]),
            dtype: Dtype::Double,
        };

        let result = tensor1.add(tensor2, vec![2, 3]);

        let expected_tensor = Tensor {
            shape: vec![2, 3],
            raw_data: vec_f64_to_raw_data(&vec![4.0, 5.0, 6.0, 5.0, 6.0, 7.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn test_add_broadcasting_non_equal_shape() {
        let tensor1 = Tensor {
            shape: vec![2, 1],
            raw_data: vec_f64_to_raw_data(&vec![1.0, 2.0]),
            dtype: Dtype::Double,
        };

        let tensor2 = Tensor {
            shape: vec![3],
            raw_data: vec_f64_to_raw_data(&vec![3.0, 4.0, 5.0]),
            dtype: Dtype::Double,
        };

        let result = tensor1.add(tensor2, vec![2, 3]);

        let expected_tensor = Tensor {
            shape: vec![2, 3],
            raw_data: vec_f64_to_raw_data(&vec![4.0, 5.0, 6.0, 5.0, 6.0, 7.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn test_add_scalar_broadcasting() {
        let tensor1 = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![1.0, 2.0, 3.0, 4.0]),
            dtype: Dtype::Double,
        };

        let tensor2 = Tensor {
            shape: vec![1],
            raw_data: vec_f64_to_raw_data(&vec![10.0]),
            dtype: Dtype::Double,
        };

        let result = tensor1.add(tensor2, vec![2, 2]);

        let expected_tensor = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![11.0, 12.0, 13.0, 14.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn test_mul_same_shape() {
        let tensor1 = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![1.0, 2.0, 3.0, 4.0]),
            dtype: Dtype::Double,
        };

        let tensor2 = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![5.0, 6.0, 7.0, 8.0]),
            dtype: Dtype::Double,
        };

        let result = tensor1.mul(tensor2, vec![2, 2]);

        let expected_tensor = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![5.0, 12.0, 21.0, 32.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn test_mul_broadcasting() {
        let tensor1 = Tensor {
            shape: vec![2, 1],
            raw_data: vec_f64_to_raw_data(&vec![1.0, 2.0]),
            dtype: Dtype::Double,
        };

        let tensor2 = Tensor {
            shape: vec![1, 3],
            raw_data: vec_f64_to_raw_data(&vec![3.0, 4.0, 5.0]),
            dtype: Dtype::Double,
        };

        let result = tensor1.mul(tensor2, vec![2, 3]);

        let expected_tensor = Tensor {
            shape: vec![2, 3],
            raw_data: vec_f64_to_raw_data(&vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn test_mul_scalar_broadcasting() {
        let tensor1 = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![1.0, 2.0, 3.0, 4.0]),
            dtype: Dtype::Double,
        };

        let tensor2 = Tensor {
            shape: vec![1],
            raw_data: vec_f64_to_raw_data(&vec![10.0]),
            dtype: Dtype::Double,
        };

        let result = tensor1.mul(tensor2, vec![2, 2]);

        let expected_tensor = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![10.0, 20.0, 30.0, 40.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn test_mod_same_shape() {
        let tensor1 = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![5.0, 6.0, 7.0, 8.0]),
            dtype: Dtype::Double,
        };

        let tensor2 = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![3.0, 4.0, 5.0, 2.0]),
            dtype: Dtype::Double,
        };

        let result = tensor1.modulo(tensor2, vec![2, 2]);

        let expected_tensor = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![2.0, 2.0, 2.0, 0.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn test_mod_broadcasting() {
        let tensor1 = Tensor {
            shape: vec![2, 1],
            raw_data: vec_f64_to_raw_data(&vec![7.0, 10.0]),
            dtype: Dtype::Double,
        };

        let tensor2 = Tensor {
            shape: vec![1, 3],
            raw_data: vec_f64_to_raw_data(&vec![3.0, 4.0, 5.0]),
            dtype: Dtype::Double,
        };

        let result = tensor1.modulo(tensor2, vec![2, 3]);

        let expected_tensor = Tensor {
            shape: vec![2, 3],
            raw_data: vec_f64_to_raw_data(&vec![1.0, 3.0, 2.0, 1.0, 2.0, 0.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn test_mod_scalar_broadcasting() {
        let tensor1 = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![9.0, 10.0, 11.0, 12.0]),
            dtype: Dtype::Double,
        };

        let tensor2 = Tensor {
            shape: vec![1],
            raw_data: vec_f64_to_raw_data(&vec![5.0]),
            dtype: Dtype::Double,
        };

        let result = tensor1.modulo(tensor2, vec![2, 2]);

        let expected_tensor = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![4.0, 0.0, 1.0, 2.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn test_lessthan_same_shape() {
        let tensor1 = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![1.0, 2.0, 3.0, 4.0]),
            dtype: Dtype::Double,
        };

        let tensor2 = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![5.0, 2.0, 1.0, 4.0]),
            dtype: Dtype::Double,
        };

        let result = tensor1.lessthan(tensor2, vec![2, 2]);

        let expected_tensor = Tensor {
            shape: vec![2, 2],
            raw_data: vec_bool_to_raw_data(&vec![true, false, false, false]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn test_lessthan_broadcasting() {
        let tensor1 = Tensor {
            shape: vec![2, 1],
            raw_data: vec_f64_to_raw_data(&vec![1.0, 2.0]),
            dtype: Dtype::Double,
        };

        let tensor2 = Tensor {
            shape: vec![1, 3],
            raw_data: vec_f64_to_raw_data(&vec![3.0, 1.0, 2.0]),
            dtype: Dtype::Double,
        };

        let result = tensor1.lessthan(tensor2, vec![2, 3]);

        let expected_tensor = Tensor {
            shape: vec![2, 3],
            raw_data: vec_bool_to_raw_data(&vec![true, false, true, true, false, false]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn test_lessthan_scalar_broadcasting() {
        let tensor1 = Tensor {
            shape: vec![2, 2],
            raw_data: vec_f64_to_raw_data(&vec![1.0, 2.0, 3.0, 4.0]),
            dtype: Dtype::Double,
        };

        let tensor2 = Tensor {
            shape: vec![1],
            raw_data: vec_f64_to_raw_data(&vec![3.0]),
            dtype: Dtype::Double,
        };

        let result = tensor1.lessthan(tensor2, vec![2, 2]);

        let expected_tensor = Tensor {
            shape: vec![2, 2],
            raw_data: vec_bool_to_raw_data(&vec![true, true, false, false]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn reduce_sum_no_keep_dims() {
        let tensor1 = Tensor {
            shape: vec![3, 2, 2],
            raw_data: vec_f64_to_raw_data(&vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ]),
            dtype: Dtype::Double,
        };

        let axes = Tensor {
            shape: vec![1],
            raw_data: vec_i32_into_raw_data(&vec![1]),
            dtype: Dtype::I32,
        };

        let result = tensor1.sum_reduce(Some(axes), Some(false), None);

        let expected_tensor = Tensor {
            shape: vec![3, 2],
            raw_data: vec_f64_to_raw_data(&vec![4.0, 6.0, 12.0, 14.0, 20.0, 22.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn reduce_sum_keep_dims() {
        let tensor1 = Tensor {
            shape: vec![3, 2, 2],
            raw_data: vec_f64_to_raw_data(&vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ]),
            dtype: Dtype::Double,
        };

        let axes = Tensor {
            shape: vec![1],
            raw_data: vec_i32_into_raw_data(&vec![1]),
            dtype: Dtype::I32,
        };

        let result = tensor1.sum_reduce(Some(axes), Some(true), None);

        let expected_tensor = Tensor {
            shape: vec![3, 1, 2],
            raw_data: vec_f64_to_raw_data(&vec![4.0, 6.0, 12.0, 14.0, 20.0, 22.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn reduce_sum_default_axes_keepdims() {
        let tensor1 = Tensor {
            shape: vec![3, 2, 2],
            raw_data: vec_f64_to_raw_data(&vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ]),
            dtype: Dtype::Double,
        };

        let result = tensor1.sum_reduce(None, Some(true), None);

        let expected_tensor = Tensor {
            shape: vec![1, 1, 1],
            raw_data: vec_f64_to_raw_data(&vec![78.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn reduce_sum_negative_axes_keepdims() {
        let tensor1 = Tensor {
            shape: vec![3, 2, 2],
            raw_data: vec_f64_to_raw_data(&vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ]),
            dtype: Dtype::Double,
        };

        let axes = Tensor {
            shape: vec![1],
            raw_data: vec_i32_into_raw_data(&vec![-2]),
            dtype: Dtype::I32,
        };

        let result = tensor1.sum_reduce(Some(axes), Some(true), None);

        let expected_tensor = Tensor {
            shape: vec![3, 1, 2],
            raw_data: vec_f64_to_raw_data(&vec![4.0, 6.0, 12.0, 14.0, 20.0, 22.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn reduce_sum_empty_axes_input_noop() {
        let tensor1 = Tensor {
            shape: vec![3, 2, 2],
            raw_data: vec_f64_to_raw_data(&vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ]),
            dtype: Dtype::Double,
        };

        let result = tensor1.sum_reduce(None, Some(true), Some(true));

        let expected_tensor = tensor1.clone();

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn reduce_max_no_keep_dims() {
        let tensor1 = Tensor {
            shape: vec![3, 2, 2],
            raw_data: vec_f64_to_raw_data(&vec![
                5.0, 1.0, 20.0, 2.0, 30.0, 1.0, 40.0, 2.0, 55.0, 1.0, 60.0, 2.0,
            ]),
            dtype: Dtype::Double,
        };

        let axes = Tensor {
            shape: vec![1],
            raw_data: vec_i32_into_raw_data(&vec![1]),
            dtype: Dtype::I32,
        };

        let result = tensor1.max_reduce(Some(axes), Some(false), None);

        let expected_tensor = Tensor {
            shape: vec![3, 2],
            raw_data: vec_f64_to_raw_data(&vec![20.0, 2.0, 40.0, 2.0, 60.0, 2.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn reduce_max_keep_dims() {
        let tensor1 = Tensor {
            shape: vec![3, 2, 2],
            raw_data: vec_f64_to_raw_data(&vec![
                5.0, 1.0, 20.0, 2.0, 30.0, 1.0, 40.0, 2.0, 55.0, 1.0, 60.0, 2.0,
            ]),
            dtype: Dtype::Double,
        };

        let axes = Tensor {
            shape: vec![1],
            raw_data: vec_i32_into_raw_data(&vec![1]),
            dtype: Dtype::I32,
        };

        let result = tensor1.max_reduce(Some(axes), Some(true), None);

        let expected_tensor = Tensor {
            shape: vec![3, 1, 2],
            raw_data: vec_f64_to_raw_data(&vec![20.0, 2.0, 40.0, 2.0, 60.0, 2.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn reduce_max_default_axes_keepdims() {
        let tensor1 = Tensor {
            shape: vec![3, 2, 2],
            raw_data: vec_f64_to_raw_data(&vec![
                5.0, 1.0, 20.0, 2.0, 30.0, 1.0, 40.0, 2.0, 55.0, 1.0, 60.0, 2.0,
            ]),
            dtype: Dtype::Double,
        };

        let result = tensor1.max_reduce(None, Some(true), None);

        let expected_tensor = Tensor {
            shape: vec![1, 1, 1],
            raw_data: vec_f64_to_raw_data(&vec![60.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn reduce_max_negative_axes_keepdims() {
        let tensor1 = Tensor {
            shape: vec![3, 2, 2],
            raw_data: vec_f64_to_raw_data(&vec![
                5.0, 1.0, 20.0, 2.0, 30.0, 1.0, 40.0, 2.0, 55.0, 1.0, 60.0, 2.0,
            ]),
            dtype: Dtype::Double,
        };

        let axes = Tensor {
            shape: vec![1],
            raw_data: vec_i32_into_raw_data(&vec![-2]),
            dtype: Dtype::I32,
        };

        let result = tensor1.max_reduce(Some(axes), Some(true), None);

        let expected_tensor = Tensor {
            shape: vec![3, 1, 2],
            raw_data: vec_f64_to_raw_data(&vec![20.0, 2.0, 40.0, 2.0, 60.0, 2.0]),
            dtype: Dtype::Double,
        };

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }

    #[test]
    fn reduce_max_empty_axes_input_noop() {
        let tensor1 = Tensor {
            shape: vec![3, 2, 2],
            raw_data: vec_f64_to_raw_data(&vec![
                5.0, 1.0, 20.0, 2.0, 30.0, 1.0, 40.0, 2.0, 55.0, 1.0, 60.0, 2.0,
            ]),
            dtype: Dtype::Double,
        };

        let result = tensor1.max_reduce(None, Some(true), Some(true));

        let expected_tensor = tensor1.clone();

        assert_eq!(result.shape, expected_tensor.shape);
        assert_eq!(result.raw_data, expected_tensor.raw_data);
    }
}
