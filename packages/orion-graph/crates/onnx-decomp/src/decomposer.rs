use onnx::{
    extractors::extract_graph,
    onnx_proto::{GraphProto, NodeProto},
};

use crate::{operators, utils::read_model};

pub fn decomp_onnx(model_path: &str) -> GraphProto {
    let buf = read_model(model_path);
    let graph = extract_graph(buf);

    let mut new_nodes: Vec<NodeProto> = Vec::new();

    for node in graph.node {
        match node.op_type.as_str() {
            "LogSoftmax" => {
                new_nodes.append(&mut operators::logsoftmax::decompose_logsoftmax(node));
            }
            "ReduceSumSquare" => {
                new_nodes
                    .append(&mut operators::reduce_sum_square::decompose_reduce_sum_square(node));
            }
            "ReduceL1" => {
                new_nodes.append(&mut operators::reduce_l1::decompose_reduce_l1(node));
            }
            "ReduceL2" => {
                new_nodes.append(&mut operators::reduce_l2::decompose_reduce_l2(node));
            }
            "ReduceLogSum" => {
                new_nodes.append(&mut operators::reduce_log_sum::decompose_reduce_log_sum(
                    node,
                ));
            }
            "ReduceLogSumExp" => {
                new_nodes
                    .append(&mut operators::reduce_log_sum_exp::decompose_reduce_log_sum_exp(node));
            }
            _ => new_nodes.push(node),
        }
    }

    GraphProto {
        node: new_nodes,
        name: graph.name.clone(),
        initializer: graph.initializer.clone(),
        sparse_initializer: graph.sparse_initializer.clone(),
        doc_string: graph.doc_string.clone(),
        input: graph.input.clone(),
        output: graph.output.clone(),
        value_info: graph.value_info.clone(),
        quantization_annotation: graph.quantization_annotation.clone(),
        metadata_props: graph.metadata_props.clone(),
    }
}
