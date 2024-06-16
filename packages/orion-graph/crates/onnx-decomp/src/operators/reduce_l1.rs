use onnx::onnx_proto::NodeProto;

pub(crate) fn decompose_reduce_l1(node: NodeProto) -> Vec<NodeProto> {
    let mut output_name = node.output.first().unwrap().clone();
    output_name.push_str("_output");

    let mut abs_name = node.name.clone();
    let mut reducesum_name = node.name.clone();

    if node.name.as_str() != "" {
        abs_name.push_str("_decomposed_abs");
        reducesum_name.push_str("_decomposed_reduce_sum");
    };

    let abs_input = vec![node.input.first().unwrap().clone()];

    let abs = NodeProto {
        input: abs_input,
        output: vec![output_name],
        name: abs_name,
        op_type: "Abs".to_string(),
        domain: node.domain.clone(),
        overload: node.overload.clone(),
        attribute: vec![],
        doc_string: node.doc_string.clone(),
        metadata_props: node.metadata_props.clone(),
    };

    let mut reduce_input = abs.output.clone();

    if node.input.len() == 2 {
        reduce_input.push(node.input.get(1).unwrap().clone());
    }

    let reduce_sum = NodeProto {
        input: reduce_input,
        output: node.output,
        name: reducesum_name,
        op_type: "ReduceSum".to_string(),
        domain: node.domain,
        overload: node.overload,
        attribute: node.attribute,
        doc_string: node.doc_string,
        metadata_props: node.metadata_props,
    };

    vec![abs, reduce_sum]
}
