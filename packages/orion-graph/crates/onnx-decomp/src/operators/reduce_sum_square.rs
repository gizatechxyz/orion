use onnx::onnx_proto::AttributeProto;
use onnx::onnx_proto::NodeProto;
use onnx::onnx_proto::TensorProto;

pub(crate) fn decompose_reduce_sum_square(node: NodeProto) -> Vec<NodeProto> {
    let mut output_cst_name = node.output.first().unwrap().clone();
    output_cst_name.push_str("_cst_output");

    let mut output_pow_name = node.output.first().unwrap().clone();
    output_pow_name.push_str("_pow_output");

    let mut constant_name = node.name.clone();
    let mut pow_name = node.name.clone();
    let mut reducesum_name = node.name.clone();

    if node.name.as_str() != "" {
        constant_name.push_str("_decomposed_cst");
        pow_name.push_str("_decomposed_pow");
        reducesum_name.push_str("_decomposed_reduce_sum");
    };

    let constant_attr = constant_attribute();

    let constant = NodeProto {
        input: vec![],
        output: vec![output_cst_name.clone()],
        name: constant_name,
        op_type: "Constant".to_string(),
        domain: node.domain.clone(),
        overload: node.overload.clone(),
        attribute: vec![constant_attr],
        doc_string: node.doc_string.clone(),
        metadata_props: node.metadata_props.clone(),
    };

    let mut pow_input = vec![node.input.first().unwrap().clone()];
    pow_input.push(output_cst_name);

    let pow = NodeProto {
        input: pow_input,
        output: vec![output_pow_name],
        name: pow_name,
        op_type: "Pow".to_string(),
        domain: node.domain.clone(),
        overload: node.overload.clone(),
        attribute: vec![],
        doc_string: node.doc_string.clone(),
        metadata_props: node.metadata_props.clone(),
    };

    let mut reduce_input = pow.output.clone();

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

    vec![constant, pow, reduce_sum]
}

fn constant_attribute() -> AttributeProto {
    let tensor = TensorProto {
        dims: vec![1],
        data_type: 1,
        segment: Option::None,
        float_data: vec![2.],
        int32_data: vec![],
        string_data: vec![],
        int64_data: vec![],
        name: "const_tensor".to_string(),
        doc_string: "".to_string(),
        raw_data: vec![],
        external_data: vec![],
        data_location: 0,
        double_data: vec![],
        uint64_data: vec![],
        metadata_props: vec![],
    };

    AttributeProto {
        name: "value".to_string(),
        ref_attr_name: "value".to_string(),
        doc_string: "".to_string(),
        r#type: 4,
        f: 0.,
        i: 0,
        s: vec![],
        t: Option::Some(tensor),
        g: Option::None,
        sparse_tensor: Option::None,
        tp: Option::None,
        floats: vec![],
        ints: vec![],
        strings: vec![],
        tensors: vec![],
        graphs: vec![],
        sparse_tensors: vec![],
        type_protos: vec![],
    }
}
