use crate::{
    helpers::{
        broadcast_shape, compute_reduced_shape, initialization_tensor_shape_buffer,
        initializer_onnx_to_primops, input_onnx_to_primops, output_onnx_to_primops,
    },
    operators,
};
use onnx::extractors::extract_graph;
use onnx_decomp::utils::read_model;
use orion::{
    helpers::vec_f64_to_raw_data,
    primgraph::{Attribute, Dtype, Initializer, PrimGraph, PrimNode, Primops, Tensor},
};
use std::{
    collections::HashMap,
    f64::consts::{LN_2, LOG2_E},
};

/// Decomposes an ONNX graph into a PrimGraph.
///
/// # Arguments
///
/// * `model_path: &str` - The path of the input ONNX model.
///
/// # Returns
///
/// * `output : PrimGraph` - A PrimGraph graph equivalent to the ONNX input graph using only PrimOps.
pub fn decomp_onnx_to_primops(model_path: &str) -> PrimGraph {
    let buf = read_model(model_path);
    let graph = extract_graph(buf);

    let mut new_nodes = vec![];
    let mut tensor_shape_buf: HashMap<String, Vec<usize>> =
        initialization_tensor_shape_buffer(graph.input.clone(), graph.initializer.clone());
    let mut initializers = initializer_onnx_to_primops(graph.initializer.clone());

    for node in graph.node {
        match node.op_type.as_str() {
            "Log" => {
                let input_dim: Vec<usize> = tensor_shape_buf
                    .get(node.clone().input.first().unwrap())
                    .unwrap()
                    .clone();

                let log2_output_name = format!("{}__log2_output", node.name);
                let new_node_log2 = PrimNode {
                    name: node.name.clone(),
                    optype: Primops::Log2,
                    inputs: node.input.clone(),
                    outputs: vec![log2_output_name.clone()],
                    attributes: vec![],
                };
                new_nodes.push(new_node_log2);

                tensor_shape_buf.insert(log2_output_name.clone(), input_dim.clone());

                let ln_2_name: String = format!("{}_ln_2_initializer", node.name);
                let ln_2_initializer = Initializer {
                    name: ln_2_name.clone(),
                    tensor: Tensor {
                        shape: vec![1],
                        raw_data: vec_f64_to_raw_data(&[LN_2]),
                        dtype: Dtype::Double,
                    },
                };
                initializers.push(ln_2_initializer);

                tensor_shape_buf.insert(ln_2_name.clone(), vec![1]);

                let new_node_mul = PrimNode {
                    name: format!("{}_mul_log2_x_and_ln_2", node.name),
                    optype: Primops::Mul,
                    inputs: vec![log2_output_name, ln_2_name],
                    outputs: node.output.clone(),
                    attributes: vec![],
                };
                new_nodes.push(new_node_mul);

                tensor_shape_buf.insert(
                    (*node.clone().output.first().unwrap()).clone(),
                    input_dim.clone(),
                );
            }
            "Exp" => {
                let input_dim: Vec<usize> = tensor_shape_buf
                    .get(node.clone().input.first().unwrap())
                    .unwrap()
                    .clone();

                let log2_e_name: String = format!("{}_log2_e_initializer", node.name);

                let log2_e_initializer = Initializer {
                    name: log2_e_name.clone(),
                    tensor: Tensor {
                        shape: vec![1],
                        raw_data: vec_f64_to_raw_data(&[LOG2_E]),
                        dtype: Dtype::Double,
                    },
                };
                initializers.push(log2_e_initializer);
                tensor_shape_buf.insert(log2_e_name.clone(), vec![1]);

                let mul_name = format!("{}_mul_output", node.name);

                let new_node_mul = PrimNode {
                    name: format!("{}_mul_x_and_log2_e", node.name),
                    optype: Primops::Mul,
                    inputs: vec![(*node.input.first().unwrap()).clone(), log2_e_name.clone()],
                    outputs: vec![mul_name.clone()],
                    attributes: vec![],
                };
                new_nodes.push(new_node_mul);

                tensor_shape_buf.insert(mul_name.clone(), input_dim.clone());

                let new_node_exp2 = PrimNode {
                    name: node.name.clone(),
                    optype: Primops::Exp2,
                    inputs: vec![mul_name, log2_e_name],
                    outputs: node.output.clone(),
                    attributes: vec![],
                };
                new_nodes.push(new_node_exp2);

                tensor_shape_buf.insert((*node.output.first().unwrap()).clone(), input_dim.clone());
            }
            "Sin" => {
                let new_node = PrimNode {
                    name: node.name.clone(),
                    optype: Primops::Sin,
                    inputs: node.input.clone(),
                    outputs: node.output.clone(),
                    attributes: vec![],
                };
                new_nodes.push(new_node);

                tensor_shape_buf.insert(
                    (*node.clone().output.first().unwrap()).clone(),
                    tensor_shape_buf
                        .get(node.clone().input.first().unwrap())
                        .unwrap()
                        .clone(),
                );
            }
            "Sqrt" => {
                let new_node = PrimNode {
                    name: node.name.clone(),
                    optype: Primops::Sqrt,
                    inputs: node.input.clone(),
                    outputs: node.output.clone(),
                    attributes: vec![],
                };
                new_nodes.push(new_node);

                tensor_shape_buf.insert(
                    (*node.clone().output.first().unwrap()).clone(),
                    tensor_shape_buf
                        .get(node.clone().input.first().unwrap())
                        .unwrap()
                        .clone(),
                );
            }
            "Reciprocal" => {
                let new_node = PrimNode {
                    name: node.name.clone(),
                    optype: Primops::Recip,
                    inputs: node.input.clone(),
                    outputs: node.output.clone(),
                    attributes: vec![],
                };
                new_nodes.push(new_node);

                tensor_shape_buf.insert(
                    (*node.clone().output.first().unwrap()).clone(),
                    tensor_shape_buf
                        .get(node.clone().input.first().unwrap())
                        .unwrap()
                        .clone(),
                );
            }
            "Add" => {
                let new_node = PrimNode {
                    name: node.name.clone(),
                    optype: Primops::Add,
                    inputs: node.input.clone(),
                    outputs: node.output.clone(),
                    attributes: vec![],
                };
                new_nodes.push(new_node);

                let shape_output = broadcast_shape(
                    tensor_shape_buf
                        .get(node.clone().input.first().unwrap())
                        .unwrap()
                        .clone(),
                    tensor_shape_buf
                        .get(node.clone().input.get(1).unwrap())
                        .unwrap()
                        .clone(),
                );
                tensor_shape_buf.insert(
                    (*node.clone().output.first().unwrap()).clone(),
                    shape_output,
                );
            }
            "Mul" => {
                let new_node = PrimNode {
                    name: node.name.clone(),
                    optype: Primops::Mul,
                    inputs: node.input.clone(),
                    outputs: node.output.clone(),
                    attributes: vec![],
                };
                new_nodes.push(new_node);

                let shape_output = broadcast_shape(
                    tensor_shape_buf
                        .get(node.clone().input.first().unwrap())
                        .unwrap()
                        .clone(),
                    tensor_shape_buf
                        .get(node.clone().input.get(1).unwrap())
                        .unwrap()
                        .clone(),
                );
                tensor_shape_buf.insert(
                    (*node.clone().output.first().unwrap()).clone(),
                    shape_output,
                );
            }
            "Mod" => {
                let new_node = PrimNode {
                    name: node.name.clone(),
                    optype: Primops::Mod,
                    inputs: node.input.clone(),
                    outputs: node.output.clone(),
                    attributes: vec![],
                };
                new_nodes.push(new_node);

                let shape_output = broadcast_shape(
                    tensor_shape_buf
                        .get(node.clone().input.first().unwrap())
                        .unwrap()
                        .clone(),
                    tensor_shape_buf
                        .get(node.clone().input.get(1).unwrap())
                        .unwrap()
                        .clone(),
                );
                tensor_shape_buf.insert(
                    (*node.clone().output.first().unwrap()).clone(),
                    shape_output,
                );
            }
            "Less" => {
                let new_node = PrimNode {
                    name: node.name.clone(),
                    optype: Primops::LessThan,
                    inputs: node.input.clone(),
                    outputs: node.output.clone(),
                    attributes: vec![],
                };
                new_nodes.push(new_node);

                let shape_output = broadcast_shape(
                    tensor_shape_buf
                        .get(node.clone().input.first().unwrap())
                        .unwrap()
                        .clone(),
                    tensor_shape_buf
                        .get(node.clone().input.get(1).unwrap())
                        .unwrap()
                        .clone(),
                );
                tensor_shape_buf.insert(
                    (*node.clone().output.first().unwrap()).clone(),
                    shape_output,
                );
            }
            "ReduceSum" => {
                let mut attributes = vec![];

                let (keepdims, noop_with_empty_axes) = match node.attribute.first() {
                    Option::Some(attr1) => match attr1.name.as_str() {
                        "keepdims" => {
                            attributes.push(Attribute {
                                name: "keepdims".to_string(),
                                value: attr1.i == 1,
                            });
                            match node.attribute.get(1) {
                                Option::Some(attr2) => {
                                    attributes.push(Attribute {
                                        name: "noop_with_empty_axes".to_string(),
                                        value: attr2.i == 1,
                                    });
                                    (Option::Some(attr1.i == 1), Option::Some(attr2.i == 1))
                                }
                                Option::None => (Option::Some(attr1.i == 1), Option::None),
                            }
                        }
                        "noop_with_empty_axes" => {
                            attributes.push(Attribute {
                                name: "noop_with_empty_axes".to_string(),
                                value: attr1.i == 1,
                            });
                            (Option::None, Option::Some(attr1.i == 1))
                        }
                        _ => panic!("wrong attribute name"),
                    },
                    Option::None => (Option::None, Option::None),
                };

                let new_node = PrimNode {
                    name: node.name.clone(),
                    optype: Primops::SumReduce,
                    inputs: node.input.clone(),
                    outputs: node.output.clone(),
                    attributes,
                };
                new_nodes.push(new_node);

                let axes = match node.input.clone().get(1) {
                    Option::Some(_) => panic!("not supported yet"),
                    Option::None => Option::None,
                };

                let shape_output = compute_reduced_shape(
                    tensor_shape_buf
                        .get(node.clone().input.first().unwrap())
                        .unwrap()
                        .clone(),
                    axes,
                    keepdims,
                    noop_with_empty_axes,
                );
                tensor_shape_buf.insert(
                    (*node.clone().output.first().unwrap()).clone(),
                    shape_output,
                );
            }
            "ReduceMax" => {
                let mut attributes = vec![];

                let (keepdims, noop_with_empty_axes) = match node.attribute.first() {
                    Option::Some(attr1) => match attr1.name.as_str() {
                        "keepdims" => {
                            attributes.push(Attribute {
                                name: "keepdims".to_string(),
                                value: attr1.i == 1,
                            });
                            match node.attribute.get(1) {
                                Option::Some(attr2) => {
                                    attributes.push(Attribute {
                                        name: "noop_with_empty_axes".to_string(),
                                        value: attr2.i == 1,
                                    });
                                    (Option::Some(attr1.i == 1), Option::Some(attr2.i == 1))
                                }
                                Option::None => (Option::Some(attr1.i == 1), Option::None),
                            }
                        }
                        "noop_with_empty_axes" => {
                            attributes.push(Attribute {
                                name: "noop_with_empty_axes".to_string(),
                                value: attr1.i == 1,
                            });
                            (Option::None, Option::Some(attr1.i == 1))
                        }
                        _ => panic!("wrong attribute name"),
                    },
                    Option::None => (Option::None, Option::None),
                };

                let new_node = PrimNode {
                    name: node.name.clone(),
                    optype: Primops::MaxReduce,
                    inputs: node.input.clone(),
                    outputs: node.output.clone(),
                    attributes,
                };
                new_nodes.push(new_node);

                let axes = match node.input.clone().get(1) {
                    Option::Some(_) => panic!("not supported yet"),
                    Option::None => Option::None,
                };

                let shape_output = compute_reduced_shape(
                    tensor_shape_buf
                        .get(node.clone().input.first().unwrap())
                        .unwrap()
                        .clone(),
                    axes,
                    keepdims,
                    noop_with_empty_axes,
                );
                tensor_shape_buf.insert(
                    (*node.clone().output.first().unwrap()).clone(),
                    shape_output,
                );
            }
            "MatMul" => {
                let (mut new_node, names, shapes, initializers_matmul) =
                    operators::matmul::decompose_matmul(
                        node.clone(),
                        tensor_shape_buf
                            .get(node.clone().input.first().unwrap())
                            .unwrap()
                            .clone(),
                        tensor_shape_buf
                            .get(node.clone().input.get(1).unwrap())
                            .unwrap()
                            .clone(),
                    );
                new_nodes.append(&mut new_node);

                for (name, shape) in names.iter().zip(shapes.iter()) {
                    tensor_shape_buf.insert((*name).clone(), (*shape).clone());
                }

                for initializer in initializers_matmul {
                    initializers.push(initializer.clone());
                    tensor_shape_buf.insert(initializer.name, initializer.tensor.shape);
                }
            }
            _ => panic!("not supported yet"),
        }
    }

    PrimGraph {
        inputs: input_onnx_to_primops(graph.input.clone()),
        outputs: output_onnx_to_primops(graph.output.clone()),
        nodes: new_nodes,
        initializers,
        shape: tensor_shape_buf,
    }
}
