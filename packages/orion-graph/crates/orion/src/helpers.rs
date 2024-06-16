use byteorder::ByteOrder;
use byteorder::{LittleEndian, WriteBytesExt};
use prost::alloc::vec::Vec;

/// Converts a vector of boolean values into raw byte data.
///
/// # Arguments
///
/// * `data: &[bool]` - A reference to a vector of boolean values.
///
/// # Returns
///
/// A vector of `u8` values where each `bool` value is converted to a byte (`0` for `false` and `1` for `true`).
///
/// # Output
///
/// * `output: Vec<u8>` - The result of converting the boolean values into raw byte data.
pub fn vec_bool_to_raw_data(data: &[bool]) -> Vec<u8> {
    data.iter().map(|&b| if b { 1 } else { 0 }).collect()
}

/// Converts raw byte data into a vector of boolean values.
///
/// # Arguments
///
/// * `data: &[u8]` - A reference to a vector of bytes representing the raw data.
///
/// # Returns
///
/// A vector of `bool` values interpreted from the input byte data. Each `u8` value is interpreted as a `bool`, where `0` is `false` and any non-zero value is `true`.
///
/// # Note
///
/// The `raw_data` is a field in `TensorProto` from ONNX, which stores data as a `Vec<u8>`. This function assumes the data is known to represent boolean values.
///
/// # Output
///
/// * `output: Vec<bool>` - The result of converting the raw byte data into `bool` values.
pub fn vec_raw_data_to_bool(data: &[u8]) -> Vec<bool> {
    data.iter().map(|&byte| byte != 0).collect()
}

/// Converts a vector of i32 values into a raw_data vector
///
/// # Arguments
///
/// * data: &Vec<i32> - The input vector containing i32 values.
///
/// # Returns
///
/// vec_i32_into_raw_data takes a vector of i32 values and returns a vector with the values converted to raw u8 data.
///
/// Note: This function assumes the data needs to be converted to u8 format in little-endian order, suitable for storing in a TensorProto from ONNX.
///
/// * output: Vec<u8> - The converted raw data vector.
pub fn vec_i32_into_raw_data(data: &Vec<i32>) -> Vec<u8> {
    let mut raw_data = Vec::with_capacity(data.len() * 4);
    for value in data {
        raw_data.write_i32::<LittleEndian>(*value).unwrap();
    }
    raw_data
}

pub fn vec_u32_into_raw_data(data: &Vec<u32>) -> Vec<u8> {
    let mut raw_data = Vec::with_capacity(data.len() * 4);
    for value in data {
        raw_data.write_u32::<LittleEndian>(*value).unwrap();
    }
    raw_data
}

/// Converts a vector of f64 values into a raw_data vector
///
/// # Arguments
///
/// * data: &[f64] - The input vector containing f64 values.
///
/// # Returns
///
/// vec_f64_to_raw_data takes a vector of f64 values and returns a vector with the values converted to raw u8 data.
///
/// Note: This function assumes the data needs to be converted to u8 format in little-endian order, suitable for storing in a TensorProto from ONNX.
///
/// * output: Vec<u8> - The converted raw data vector.
pub fn vec_f64_to_raw_data(data: &[f64]) -> Vec<u8> {
    let mut raw_data = Vec::with_capacity(data.len() * 8);
    for value in data {
        raw_data.write_f64::<LittleEndian>(*value).unwrap();
    }
    raw_data
}

/// Converts a raw_data vector into a vector of f64 values
///
/// # Arguments
///
/// * `raw_data: &[u8]` - The raw_data input.
///
/// # Returns
///
/// `vec_raw_data_to_f64` takes a raw_data Vec as an input and returns a vector with the values interpreted as `f64`.
///
/// Note : `raw_data` is a field in `TensorProto` from ONNX, which stores data as a `Vec<u8>`. This function assumes the data is of type `f64`.
///
/// * `output : Vec<f64>` - converted vector
pub fn vec_raw_data_to_f64(raw_data: &[u8]) -> Vec<f64> {
    raw_data
        .chunks_exact(8)
        .map(LittleEndian::read_f64)
        .collect()
}

/// Converts a raw_data vector into a vector of f64 values
///
/// # Arguments
///
/// * `raw_data: &[u8]` - The raw_data input.
///
/// # Returns
///
/// `vec_raw_data_to_i32` takes a raw_data Vec as an input and returns a vector with the values interpreted as `i32`.
///
/// Note : `raw_data` is a field in `TensorProto` from ONNX, which stores data as a `Vec<u8>`. This function assumes the data is of type `i32`.
///
/// * `output : Vec<i32>` - converted vector
pub fn vec_raw_data_to_i32(raw_data: &[u8]) -> Vec<i32> {
    raw_data
        .chunks_exact(4)
        .map(LittleEndian::read_i32)
        .collect()
}

pub fn vec_raw_data_to_u32(raw_data: &[u8]) -> Vec<u32> {
    raw_data
        .chunks_exact(4)
        .map(LittleEndian::read_u32)
        .collect()
}

pub(crate) fn broadcast_index_mapping(shape: Vec<usize>, indices: Vec<usize>) -> usize {
    if shape.len() == indices.len() {
        broadcast_index_mapping_equal_shape(shape, indices)
    } else {
        broadcast_index_mapping_non_equal_shape(shape, indices)
    }
}

fn broadcast_index_mapping_equal_shape(shape: Vec<usize>, indices: Vec<usize>) -> usize {
    let mut result = 0_usize;
    let stride = stride(shape.clone());

    for ((shape_val, indices_val), stride_val) in shape.iter().zip(indices).zip(stride) {
        let index = (indices_val % shape_val) * stride_val;
        result += index;
    }

    result
}

fn broadcast_index_mapping_non_equal_shape(
    mut shape: Vec<usize>,
    mut indices: Vec<usize>,
) -> usize {
    let mut result = 0_usize;
    let mut stride = stride(shape.clone());

    let mut offset = if shape.len() > indices.len() {
        shape.len() - indices.len()
    } else {
        0
    };

    while let Option::Some(shape_val) = shape.pop() {
        let stride_val = stride.pop().unwrap_or(1);

        let index_val = if offset > 0 {
            offset -= 1;
            0
        } else {
            indices.pop().unwrap_or(0)
        };

        let index = (index_val % shape_val) * stride_val;
        result += index;
    }

    result
}

fn stride(mut shape: Vec<usize>) -> Vec<usize> {
    let mut strides = vec![];
    let mut stride = 1;

    while let Option::Some(shape_val) = shape.pop() {
        strides.push(stride);
        stride *= shape_val;
    }
    strides.reverse();

    strides
}

pub(crate) fn len_from_shape(shape: &[usize]) -> usize {
    shape.iter().product()
}

pub(crate) fn unravel_index(index: usize, shape: &[usize]) -> Vec<usize> {
    assert!(!shape.is_empty(), "shape cannot be empty");

    let mut result = vec![];
    let mut remainder = index;
    let mut stride = len_from_shape(shape);

    for i in shape {
        stride /= i;
        let coord = remainder / stride;

        remainder %= stride;
        result.push(coord);
    }

    result
}

pub(crate) fn sum_along_axis(data: &[f64], shape: &[usize], axis: usize) -> Vec<f64> {
    let mut result = Vec::new();
    let stride = shape.iter().skip(axis + 1).product::<usize>();
    let num_outer_dims = shape.iter().take(axis).product::<usize>();
    let axis_size = shape[axis];
    let axis_stride = axis_size * stride;

    for outer_idx in 0..num_outer_dims {
        let outer_offset = outer_idx * axis_stride;
        for inner_idx in 0..stride {
            let mut sum = 0.0;
            for axis_idx in 0..axis_size {
                sum += data[outer_offset + axis_idx * stride + inner_idx];
            }
            result.push(sum);
        }
    }

    result
}

pub(crate) fn max_along_axis(data: &[f64], shape: &[usize], axis: usize) -> Vec<f64> {
    let mut result = Vec::new();
    let stride = shape.iter().skip(axis + 1).product::<usize>();
    let num_outer_dims = shape.iter().take(axis).product::<usize>();
    let axis_size = shape[axis];
    let axis_stride = axis_size * stride;

    for outer_idx in 0..num_outer_dims {
        let outer_offset = outer_idx * axis_stride;
        for inner_idx in 0..stride {
            let mut max_value = f64::MIN;
            for axis_idx in 0..axis_size {
                let value = data[outer_offset + axis_idx * stride + inner_idx];
                if value > max_value {
                    max_value = value;
                }
            }
            result.push(max_value);
        }
    }

    result
}
