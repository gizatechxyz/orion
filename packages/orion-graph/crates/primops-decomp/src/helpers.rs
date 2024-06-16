use onnx::onnx_proto::{
    tensor_shape_proto::dimension::Value, type_proto, TensorProto, ValueInfoProto,
};
use orion::primgraph::{Dtype, Initializer, Input, Output, Tensor};
use std::cmp::max;
use std::collections::HashMap;

/// Initializes the tensor_shape_buffer with the inputs and initalizers of the ONNX model
///
/// # Arguments
///
/// * `input_onnx: Vec<ValueInfoProto>` - inputs field of ONNX graph.
/// * `initializers_onnx: Vec<TensorProto>` - initializers field of ONNX graph.
///
/// # Returns
///
/// * `output : HashMap<String, Vec<usize>>` - tensor_shape_buffer, maps each tensor name to their Vec<usize> shapes.
pub(crate) fn initialization_tensor_shape_buffer(
    input_onnx: Vec<ValueInfoProto>,
    initializers_onnx: Vec<TensorProto>,
) -> HashMap<String, Vec<usize>> {
    let mut tensor_shape_buffer: HashMap<String, Vec<usize>> = HashMap::new();

    for input in input_onnx {
        let name = input.clone().name;

        let shape = match input.r#type.clone().unwrap().value.unwrap() {
            type_proto::Value::TensorType(tensor) => {
                let mut dim_u32 = vec![];
                for d in tensor.shape.unwrap().dim {
                    match d.value.unwrap() {
                        Value::DimValue(val) => dim_u32.push(val.try_into().unwrap()),
                        Value::DimParam(_) => panic!("not supported"),
                    }
                }
                dim_u32
            }
            _ => panic!("not supported"),
        };

        tensor_shape_buffer.insert(name, shape);
    }

    for initializer in initializers_onnx {
        let name = initializer.clone().name;

        let mut shape = vec![];
        for dim in initializer.dims {
            shape.push(dim.try_into().unwrap());
        }

        tensor_shape_buffer.insert(name, shape);
    }
    tensor_shape_buffer
}

/// Computes the broadcasted shape of two tensors.
///
/// # Arguments
/// * `mut shape1: Vec<usize>` - A Vec containing the shape of the first tensor as usize elements.
/// * `mut shape2: Vec<usize>` - A Vec containing the shape of the second tensor as usize elements.
///
/// # Panics
/// * Panics if the shapes of the tensors are not compatible.
///
/// # Returns
/// * A Vec of usize representing the broadcasted shape.
pub(crate) fn broadcast_shape(mut shape1: Vec<usize>, mut shape2: Vec<usize>) -> Vec<usize> {
    check_compatibility(shape1.clone(), shape2.clone());
    let mut result = vec![];

    while !shape1.is_empty() || !shape2.is_empty() {
        let dim1 = shape1.pop().unwrap_or(1);
        let dim2 = shape2.pop().unwrap_or(1);

        let broadcasted_dim = max(dim1, dim2);
        result.push(broadcasted_dim);
    }
    result.reverse();
    result
}

/// Checks if two tensor shapes are compatible for broadcasting.
///
/// # Arguments
/// * `shape_1` - A Vec containing the first tensor's shape as usize elements.
/// * `shape_2` - A Vec containing the second tensor's shape as usize elements.
///
/// # Panics
/// * Panics if the shapes are not compatible for broadcasting.
fn check_compatibility(shape_1: Vec<usize>, shape_2: Vec<usize>) {
    let mut iter_1 = shape_1.len();
    let mut iter_2 = shape_2.len();

    while iter_1 > 0 || iter_2 > 0 {
        let dim_1 = if iter_1 > 0 { shape_1[iter_1 - 1] } else { 1 };
        let dim_2 = if iter_2 > 0 { shape_2[iter_2 - 1] } else { 1 };

        if dim_1 != dim_2 && dim_1 != 1 && dim_2 != 1 {
            panic!("tensors shape must match")
        }

        iter_1 = iter_1.saturating_sub(1);
        iter_2 = iter_2.saturating_sub(1);
    }
}

/// Computes the reduced shape of two tensors (ie. the shape obtained after performing the operation SumReduce or MaxReduce)
///
/// # Arguments
/// * `mut shape1: Vec<usize>` - A Vec containing the shape of the first tensor as usize elements.
/// * `mut shape2: Vec<usize>` - A Vec containing the shape of the second tensor as usize elements.
///
/// # Panics
/// * Panics if the shapes of the tensors are not compatible.
///
/// # Returns
/// * A Vec of usize representing the reduce shape.
pub(crate) fn compute_reduced_shape(
    shape_input: Vec<usize>,
    axes: Option<Vec<i32>>,
    keepdims: Option<bool>,
    noop_with_empty_axes: Option<bool>,
) -> Vec<usize> {
    let noop = noop_with_empty_axes.unwrap_or(false);

    let axes: Vec<usize> = match axes {
        Some(axes) => axes
            .iter()
            .map(|&axis| {
                if axis < 0 {
                    (shape_input.len() as i32 + axis) as usize
                } else {
                    axis as usize
                }
            })
            .collect(),
        None => {
            if noop {
                return shape_input;
            }
            (0..shape_input.len()).collect()
        }
    };

    let keepdims = keepdims.unwrap_or(false);

    let mut result_shape = shape_input;

    for &axis in axes.iter().rev() {
        if keepdims {
            result_shape[axis] = 1;
        } else {
            result_shape.remove(axis);
        }
    }
    result_shape
}

/// Converts ONNX input type Vec<ValueInfoProto> to PrimGraph input type Vec<Input>
///
/// # Arguments
///
/// * `input_onnx: Vec<ValueInfoProto>` - input field of ONNX graph.
///
/// # Returns
///
/// * `output : Vec<Input>` - input field of Primgraph
pub(crate) fn input_onnx_to_primops(input_onnx: Vec<ValueInfoProto>) -> Vec<Input> {
    let mut input_primops = vec![];

    for input in input_onnx {
        let name = input.clone().name;

        let dtype = match input.clone().r#type.clone().unwrap().value.unwrap() {
            type_proto::Value::TensorType(tensor) => {
                if tensor.elem_type == 11 {
                    Dtype::Double
                } else if tensor.elem_type == 6 {
                    Dtype::I32
                } else if tensor.elem_type == 9 {
                    Dtype::Bool
                } else {
                    panic!("not supported")
                }
            }
            _ => panic!("not supported"),
        };

        let dim = match input.r#type.clone().unwrap().value.unwrap() {
            type_proto::Value::TensorType(tensor) => {
                let mut dim_u32 = vec![];
                for d in tensor.shape.unwrap().dim {
                    match d.value.unwrap() {
                        Value::DimValue(val) => dim_u32.push(val.try_into().unwrap()),
                        Value::DimParam(_) => panic!("not supported"),
                    }
                }
                dim_u32
            }
            _ => panic!("not supported"),
        };

        let new_input = Input { name, dtype, dim };

        input_primops.push(new_input);
    }
    input_primops
}

/// Converts ONNX ouput type Vec<ValueInfoProto> to PrimGraph ouput type Vec<Ouput>
///
/// # Arguments
///
/// * `output_onnx: Vec<ValueInfoProto>` - ouput field of ONNX graph.
///
/// # Returns
///
/// * `output : Vec<Ouput>` - ouput field of Primgraph
pub(crate) fn output_onnx_to_primops(output_onnx: Vec<ValueInfoProto>) -> Vec<Output> {
    let mut output_primops = vec![];

    for output in output_onnx {
        let name = output.clone().name;

        let dtype = match output.clone().r#type.clone().unwrap().value.unwrap() {
            type_proto::Value::TensorType(tensor) => {
                if tensor.elem_type == 11 {
                    Dtype::Double
                } else if tensor.elem_type == 6 {
                    Dtype::I32
                } else if tensor.elem_type == 9 {
                    Dtype::Bool
                } else {
                    panic!("not supported")
                }
            }
            _ => panic!("not supported"),
        };

        let dim = match output.r#type.clone().unwrap().value.unwrap() {
            type_proto::Value::TensorType(tensor) => {
                let mut dim_u32 = vec![];
                for d in tensor.shape.unwrap().dim {
                    match d.value.unwrap() {
                        Value::DimValue(val) => dim_u32.push(val.try_into().unwrap()),
                        Value::DimParam(_) => panic!("not supported"),
                    }
                }
                dim_u32
            }
            _ => panic!("not supported"),
        };

        let new_output = Output { name, dtype, dim };

        output_primops.push(new_output);
    }
    output_primops
}

/// Converts ONNX initializer type Vec<TensorProto> to PrimGraph initializer type Vec<Initializer>
///
/// # Arguments
///
/// * `initializers_onnx: Vec<TensorProto>` - initializer field of ONNX graph.
///
/// # Returns
///
/// * `output : Vec<Initializer>` - initializer field of Primgraph
pub(crate) fn initializer_onnx_to_primops(initializers_onnx: Vec<TensorProto>) -> Vec<Initializer> {
    let mut initializers_primops = vec![];

    for initializers in initializers_onnx {
        let raw_data = initializers.clone().raw_data;

        let dtype = if initializers.clone().data_type == 11 {
            Dtype::Double
        } else if initializers.clone().data_type == 6 {
            Dtype::I32
        } else if initializers.clone().data_type == 9 {
            Dtype::Bool
        } else {
            panic!("not supported")
        };

        let mut shape = vec![];
        for dim in initializers.dims {
            shape.push(dim.try_into().unwrap());
        }

        let new_initializers = Initializer {
            name: initializers.name,
            tensor: Tensor {
                shape,
                raw_data,
                dtype,
            },
        };

        initializers_primops.push(new_initializers);
    }
    initializers_primops
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multidirectional_broadcasting_1() {
        let shape_1 = vec![2, 3, 4, 5];
        let shape_2 = vec![];

        let result = broadcast_shape(shape_1, shape_2);

        let expected_result = vec![2, 3, 4, 5];

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_multidirectional_broadcasting_2() {
        let shape_1 = vec![2, 3, 4, 5];
        let shape_2 = vec![5];

        let result = broadcast_shape(shape_1, shape_2);

        let expected_result = vec![2, 3, 4, 5];

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_multidirectional_broadcasting_3() {
        let shape_1 = vec![4, 5];
        let shape_2 = vec![2, 3, 4, 5];

        let result = broadcast_shape(shape_1, shape_2);

        let expected_result = vec![2, 3, 4, 5];

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_multidirectional_broadcasting_4() {
        let shape_1 = vec![1, 4, 5];
        let shape_2 = vec![2, 3, 1, 1];

        let result = broadcast_shape(shape_1, shape_2);

        let expected_result = vec![2, 3, 4, 5];

        assert_eq!(result, expected_result);
    }

    #[test]
    fn test_multidirectional_broadcasting_5() {
        let shape_1 = vec![3, 4, 5];
        let shape_2 = vec![2, 1, 1, 1];

        let result = broadcast_shape(shape_1, shape_2);

        let expected_result = vec![2, 3, 4, 5];

        assert_eq!(result, expected_result);
    }

    #[test]
    #[should_panic]
    fn test_multidirectional_broadcasting_should_panic_1() {
        let shape_1 = vec![3, 4];
        let shape_2 = vec![3];

        broadcast_shape(shape_1, shape_2);
    }

    #[test]
    #[should_panic]
    fn test_multidirectional_broadcasting_should_panic_2() {
        let shape_1 = vec![3, 4];
        let shape_2 = vec![1, 3];

        broadcast_shape(shape_1, shape_2);
    }
}
