pub mod encoders;
pub mod extractors;

pub mod onnx_proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}
