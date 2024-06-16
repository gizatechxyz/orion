use std::{
    fs::File,
    io::{Read, Write},
};

use onnx::{encoders::encode_model, extractors::extract_model, onnx_proto::ModelProto};

use crate::decomposer::decomp_onnx;

pub fn read_model(model_path: &str) -> Vec<u8> {
    let mut file = File::open(model_path).expect("Failed to open ONNX file");
    let mut buf = Vec::new();
    file.read_to_end(&mut buf)
        .expect("Failed to read ONNX file");

    buf
}

pub(crate) fn write_model(model_path: &str, model: &ModelProto) -> std::io::Result<()> {
    let mut file = File::create(model_path)?;
    let buf = encode_model(model);
    file.write_all(&buf)?;
    Ok(())
}

pub fn prepare_test(input_name: &str, output_name: &str) {
    let buf = read_model(format!("tests/models/{input_name}").as_str());
    let model = extract_model(buf);
    let new_graph = decomp_onnx(format!("tests/models/{input_name}").as_str());
    let new_model = ModelProto {
        ir_version: model.ir_version,
        opset_import: model.opset_import,
        producer_name: model.producer_name,
        producer_version: model.producer_version,
        domain: model.domain,
        model_version: model.model_version,
        doc_string: model.doc_string,
        graph: Option::Some(new_graph),
        metadata_props: model.metadata_props,
        training_info: model.training_info,
        functions: model.functions,
    };
    write_model(format!("tests/models/{output_name}").as_str(), &new_model)
        .expect("Failed to write ONNX file");
}
