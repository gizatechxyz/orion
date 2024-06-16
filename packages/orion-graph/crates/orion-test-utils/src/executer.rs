use crate::helpers::{initialization_tensor_buf, tensor_buf_to_output};
use orion::helpers::vec_raw_data_to_u32;
use orion::primgraph::{PrimGraph, Primops, Tensor};
use orion::primops::PrimopsTrait;

/// Executes a PrimGraph.
///
/// # Arguments
///
/// * `graph: &PrimGraph` - The graph.
/// * `inputs: Vec<Tensor>` - The inputs of the execution.
///
/// Note : Inputs must be in the same order in `graph.inputs` and in the `inputs` field of `execute_primgraph`
///
/// # Returns
///
/// `execute_primgraph`, takes a `PrimGraph` and the inputs of the graph. It then browses the graph and executes the operation on the input, the browsing follows the same topological order as the execution of ONNX graph. Then it returns the ouput of the execution.
///
/// * `output : Vec<Tensor>` - result of the execution of the graph on the inputs.
pub(crate) fn execute_primgraph(graph: &PrimGraph, inputs: Vec<Tensor>) -> Vec<Tensor> {
    let mut tensor_buf = initialization_tensor_buf(&graph.inputs, &graph.initializers, &inputs);

    for node in &graph.nodes {
        match node.optype {
            Primops::Log2 => {
                let input_log2 = tensor_buf.get(node.inputs.first().unwrap()).unwrap();
                let output_log2 = input_log2.log2();
                tensor_buf.insert(node.outputs.first().unwrap().clone(), output_log2);
            }
            Primops::Exp2 => {
                let input_exp2 = tensor_buf.get(node.inputs.first().unwrap()).unwrap();
                let output_exp2 = input_exp2.exp2();
                tensor_buf.insert(node.outputs.first().unwrap().clone(), output_exp2);
            }
            Primops::Sin => {
                let input_sin = tensor_buf.get(node.inputs.first().unwrap()).unwrap();
                let output_sin = input_sin.sin();
                tensor_buf.insert(node.outputs.first().unwrap().clone(), output_sin);
            }
            Primops::Sqrt => {
                let input_sqrt = tensor_buf.get(node.inputs.first().unwrap()).unwrap();
                let output_sqrt = input_sqrt.sqrt();
                tensor_buf.insert(node.outputs.first().unwrap().clone(), output_sqrt);
            }
            Primops::Recip => {
                let input_recip = tensor_buf.get(node.inputs.first().unwrap()).unwrap();
                let output_recip = input_recip.recip();
                tensor_buf.insert(node.outputs.first().unwrap().clone(), output_recip);
            }
            Primops::Add => {
                let input_a = tensor_buf.get(node.inputs.first().unwrap()).unwrap();
                let input_b = tensor_buf.get(node.inputs.get(1).unwrap()).unwrap();
                let output_add = input_a.add(
                    (*input_b).clone(),
                    (*graph.shape.get(node.outputs.first().unwrap()).unwrap()).clone(),
                );
                tensor_buf.insert(node.outputs.first().unwrap().clone(), output_add);
            }
            Primops::Mul => {
                let input_a = tensor_buf.get(node.inputs.first().unwrap()).unwrap();
                let input_b = tensor_buf.get(node.inputs.get(1).unwrap()).unwrap();
                let output_mul = input_a.mul(
                    (*input_b).clone(),
                    (*graph.shape.get(node.outputs.first().unwrap()).unwrap()).clone(),
                );
                tensor_buf.insert(node.outputs.first().unwrap().clone(), output_mul);
            }
            Primops::Mod => {
                let input_a = tensor_buf.get(node.inputs.first().unwrap()).unwrap();
                let input_b = tensor_buf.get(node.inputs.get(1).unwrap()).unwrap();
                let output_mod = input_a.modulo(
                    (*input_b).clone(),
                    (*graph.shape.get(node.outputs.first().unwrap()).unwrap()).clone(),
                );
                tensor_buf.insert(node.outputs.first().unwrap().clone(), output_mod);
            }
            Primops::LessThan => {
                let input_a = tensor_buf.get(node.inputs.first().unwrap()).unwrap();
                let input_b = tensor_buf.get(node.inputs.get(1).unwrap()).unwrap();
                let output_mod = input_a.lessthan(
                    (*input_b).clone(),
                    (*graph.shape.get(node.outputs.first().unwrap()).unwrap()).clone(),
                );
                tensor_buf.insert(node.outputs.first().unwrap().clone(), output_mod);
            }
            Primops::SumReduce => {
                let input = tensor_buf.get(node.inputs.first().unwrap()).unwrap();

                let (keepdims, noop_with_empty_axes) = match node.attributes.first() {
                    Option::Some(attr1) => match attr1.name.as_str() {
                        "keepdims" => match node.attributes.get(1) {
                            Option::Some(attr2) => {
                                (Option::Some(attr1.value), Option::Some(attr2.value))
                            }
                            Option::None => (Option::Some(attr1.value), Option::None),
                        },
                        "noop_with_empty_axes" => (Option::None, Option::Some(attr1.value)),
                        _ => panic!("wrong attribute name"),
                    },
                    Option::None => (Option::None, Option::None),
                };

                let axes = node
                    .inputs
                    .clone()
                    .get(1)
                    .map(|axes_name| (*tensor_buf.get(axes_name).unwrap()).clone());

                let output = input.sum_reduce(axes, keepdims, noop_with_empty_axes);
                tensor_buf.insert(node.outputs.first().unwrap().clone(), output);
            }
            Primops::MaxReduce => {
                let input = tensor_buf.get(node.inputs.first().unwrap()).unwrap();

                let (keepdims, noop_with_empty_axes) = match node.attributes.first() {
                    Option::Some(attr1) => match attr1.name.as_str() {
                        "keepdims" => match node.attributes.get(1) {
                            Option::Some(attr2) => {
                                (Option::Some(attr1.value), Option::Some(attr2.value))
                            }
                            Option::None => (Option::Some(attr1.value), Option::None),
                        },
                        "noop_with_empty_axes" => (Option::None, Option::Some(attr1.value)),
                        _ => panic!("wrong attribute name"),
                    },
                    Option::None => (Option::None, Option::None),
                };

                let axes = match node.inputs.clone().get(1) {
                    Option::Some(_) => panic!("not supported yet"),
                    Option::None => Option::None,
                };

                let output = input.max_reduce(axes, keepdims, noop_with_empty_axes);
                tensor_buf.insert(node.outputs.first().unwrap().clone(), output);
            }
            Primops::Reshape => {
                let input_data = tensor_buf.get(node.inputs.first().unwrap()).unwrap();
                let input_shape = tensor_buf.get(node.inputs.get(1).unwrap()).unwrap();

                let shape = vec_raw_data_to_u32(&input_shape.raw_data).iter().map(|&x| x as usize).collect();

                let reshaped_tensor = Tensor {
                    shape,
                    raw_data: input_data.raw_data.clone(),
                    dtype: input_data.dtype.clone(),
                };
                tensor_buf.insert(node.outputs.first().unwrap().clone(), reshaped_tensor);
            }
        }
    }

    tensor_buf_to_output(tensor_buf, &graph.outputs)
}
