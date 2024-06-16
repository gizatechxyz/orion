use onnx::onnx_proto::NodeProto;

pub(crate) fn decompose_reduce_log_sum_exp(node: NodeProto) -> Vec<NodeProto> {
    let mut output_exp_name = node.output.first().unwrap().clone();
    output_exp_name.push_str("_exp_output");

    let mut output_reduce_sum_name = node.output.first().unwrap().clone();
    output_reduce_sum_name.push_str("_reduce_sum_output");

    let mut exp_name = node.name.clone();
    let mut reduce_sum_name = node.name.clone();
    let mut log_name = node.name.clone();

    if node.name.as_str() != "" {
        exp_name.push_str("_decomposed_exp");
        reduce_sum_name.push_str("_decomposed_reduce_sum");
        log_name.push_str("_decomposed_log");
    };

    let exp_input = vec![node.input.first().unwrap().clone()];

    let exp = NodeProto {
        input: exp_input,
        output: vec![output_exp_name],
        name: exp_name,
        op_type: "Exp".to_string(),
        domain: node.domain.clone(),
        overload: node.overload.clone(),
        attribute: vec![],
        doc_string: node.doc_string.clone(),
        metadata_props: node.metadata_props.clone(),
    };

    let mut reduce_input = exp.output.clone();

    if node.input.len() == 2 {
        reduce_input.push(node.input.get(1).unwrap().clone());
    }

    let reduce_sum = NodeProto {
        input: reduce_input,
        output: vec![output_reduce_sum_name],
        name: reduce_sum_name,
        op_type: "ReduceSum".to_string(),
        domain: node.domain.clone(),
        overload: node.overload.clone(),
        attribute: node.attribute.clone(),
        doc_string: node.doc_string.clone(),
        metadata_props: node.metadata_props.clone(),
    };

    let log = NodeProto {
        input: reduce_sum.output.clone(),
        output: node.output,
        name: log_name,
        op_type: "Log".to_string(),
        domain: node.domain,
        overload: node.overload,
        attribute: vec![],
        doc_string: node.doc_string,
        metadata_props: node.metadata_props,
    };

    vec![exp, reduce_sum, log]
}
