use onnx::onnx_proto::NodeProto;

pub(crate) fn decompose_logsoftmax(node: NodeProto) -> Vec<NodeProto> {
    let mut output_name = node.output.first().unwrap().clone();
    output_name.push_str("_softmax_output");

    let mut soft_name = node.name.clone();
    let mut log_name = node.name.clone();

    if node.name.as_str() != "" {
        soft_name.push_str("_decomposed_softmax");
        log_name.push_str("_decomposed_log");
    };

    let softmax = NodeProto {
        input: node.input.clone(),
        output: vec![output_name],
        name: soft_name,
        op_type: "Softmax".to_string(),
        domain: node.domain.clone(),
        overload: node.overload.clone(),
        attribute: node.attribute.clone(),
        doc_string: node.doc_string.clone(),
        metadata_props: node.metadata_props.clone(),
    };

    let log = NodeProto {
        input: softmax.output.clone(),
        output: node.output,
        name: log_name,
        op_type: "Log".to_string(),
        domain: node.domain,
        overload: node.overload,
        attribute: vec![],
        doc_string: node.doc_string,
        metadata_props: node.metadata_props,
    };

    vec![softmax, log]
}
