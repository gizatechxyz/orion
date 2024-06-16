use crate::onnx_proto::{GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto};
use prost::Message;

pub fn extract_model(buf: Vec<u8>) -> ModelProto {
    let model = ModelProto::decode(&*buf).expect("Failed to decode ONNX file");

    model.to_owned()
}

pub fn extract_graph(buf: Vec<u8>) -> GraphProto {
    let model = ModelProto::decode(&*buf).expect("Failed to decode ONNX file");
    let graph = model.graph.as_ref().expect("Model should have a graph");

    graph.to_owned()
}

pub fn extract_initializers(graph: &GraphProto) -> Vec<TensorProto> {
    graph.initializer.to_vec()
}

pub fn extract_nodes(graph: &GraphProto) -> Vec<NodeProto> {
    graph.node.to_vec()
}

pub fn extract_graph_inputs(graph: &GraphProto) -> Vec<ValueInfoProto> {
    graph.input.to_vec()
}
