use orion::primgraph::Output;
use std::collections::HashMap;

use orion::primgraph::{Initializer, Input, Tensor};

/// Initializes the tensor buffer with the inputs of the graph and the inputs and initalizers of the model
///
/// # Arguments
///
/// * `primgraph_inputs: &[Input]` - inputs field of PrimGraph. Contains the names of the inputs.
/// * `initializers: &Vec<Initializer>` - initializers field of the PrimGraph.
/// * `primops_inputs: &[Tensor]` - inputs of graph. Contains the data of the inputs.
///
/// Note : Assume that the `primops_inputs` are in the same order that in `primgraph_inputs`.
///
/// # Returns
///
/// A tensor_buf that maps each tensor name to the corresponding `Tensor`.
///
/// * `output : HashMap<String, Tensor>` - tensor_buf
pub(crate) fn initialization_tensor_buf(
    primgraph_inputs: &[Input],
    initializers: &Vec<Initializer>,
    primops_inputs: &[Tensor],
) -> HashMap<String, Tensor> {
    assert!(
        primgraph_inputs.len() == primops_inputs.len(),
        "wrong number of inputs"
    );
    let mut tensor_buf: HashMap<String, Tensor> = HashMap::new();

    for (primgraph_input, primops_input) in primgraph_inputs.iter().zip(primops_inputs.iter()) {
        tensor_buf.insert(primgraph_input.name.clone(), primops_input.clone());
    }

    for initializer in initializers {
        tensor_buf.insert(initializer.name.clone(), initializer.tensor.clone());
    }
    tensor_buf
}

/// Gets a vector with the outputs tensors from tensor_buf
///
/// # Arguments
///
/// * `tensor_buf: HashMap<String, primops::Tensor>` - tensor_buf
/// * `primgraph_outputs: &Vec<Output>` - outputs of a PrimGraph
///
/// # Returns
///
/// Gets the outputs in `tensor_buf` from the names of the outputs in `primgraph_outputs`.
///
/// * `output : Vec<primops::Tensor>>` - output
pub(crate) fn tensor_buf_to_output(
    tensor_buf: HashMap<String, Tensor>,
    primgraph_outputs: &Vec<Output>,
) -> Vec<Tensor> {
    let mut result = vec![];

    for output in primgraph_outputs {
        result.push((*tensor_buf.get(&output.name).unwrap()).clone());
    }
    result
}

/// Computes a new shape for a tensor given an original shape and a new shape specification, which may include one negative dimension.
///
/// # Arguments
///
/// * `original_shape: Vec<usize>` - The original shape of the tensor.
/// * `new_shape: Vec<i32>` - The desired new shape of the tensor. One dimension can be negative, which will be inferred.
///
/// # Returns
///
/// Computes the new shape based on the original shape and the new shape specification.
///
/// * `output : Vec<usize>` - The computed new shape.
pub(crate) fn compute_new_shape(original_shape: Vec<usize>, new_shape: Vec<i32>) -> Vec<usize> {
    let total_elements: usize = original_shape.iter().product();

    let negative_dims: Vec<_> = new_shape.iter().filter(|&&dim| dim < 0).collect();

    if negative_dims.len() > 1 {
        panic!("Only one dimension can be negative.")
    }
    let mut computed_shape: Vec<usize> = new_shape
        .iter()
        .map(|&dim| {
            if dim >= 0 {
                usize::try_from(dim).unwrap()
            } else {
                0
            }
        })
        .collect();

    if negative_dims.first().is_some() {
        let known_product: usize = computed_shape.iter().filter(|&&dim| dim != 0).product();

        let inferred_dim = total_elements / known_product;

        for dim in &mut computed_shape {
            if *dim == 0 {
                *dim = inferred_dim;
            }
        }
    }

    if total_elements != computed_shape.iter().product() {
        panic!("Incompatible shapes for reshape")
    }
    computed_shape
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_new_shape() {
        let original_shape = vec![2, 3];
        let new_shape = vec![3, 2];
        let result = compute_new_shape(original_shape.clone(), new_shape);
        assert_eq!(result, vec![3, 2]);

        let original_shape = vec![2, 3];
        let new_shape = vec![3, -1];
        let result = compute_new_shape(original_shape.clone(), new_shape);
        assert_eq!(result, vec![3, 2]);

        let original_shape = vec![4, 4];
        let new_shape = vec![2, -1, 2];
        let result = compute_new_shape(original_shape.clone(), new_shape);
        assert_eq!(result, vec![2, 4, 2]);
    }
}
