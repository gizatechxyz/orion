use crate::onnx_proto::ModelProto;
use prost::Message;

pub fn encode_model(model: &ModelProto) -> Vec<u8> {
    ModelProto::encode_to_vec(model)
}
