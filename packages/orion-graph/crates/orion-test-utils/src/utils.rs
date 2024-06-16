use crate::executer::execute_primgraph;
use orion::helpers::{vec_raw_data_to_bool, vec_raw_data_to_f64, vec_raw_data_to_i32, vec_raw_data_to_u32};
use orion::primgraph::{Dtype, Tensor};
use primops_decomp::decomposer::decomp_onnx_to_primops;

use std::{
    fs::File,
    io::{Result, Write},
};

/// Writes tensors shape and data in .txt files
///
/// # Arguments
///
/// * `tensor_buf: HashMap<String, primops::Tensor>` - tensor_buf
/// * `primgraph_outputs: &Vec<Output>` - outputs of a PrimGraph
///
/// # Returns
///
/// `write_result` takes the result of a test execution and writes in .txt files, writing the shapes in one line then the data in another for each tensor in the input vector.  
///
/// * `output : Result<()>`
///
/// # Example
///
/// input : vec![Tensor {
///             shape: vec![2, 3],
///             data: vec![3.0, 4.0, 5.0, 6.0, 8.0, 10.0],
///         }]
///
/// It will produce a txt file containing
/// 2 3
/// 3.0 4.0 5.0 6.0 8.0 10.0
///
fn write_result(tensors: Vec<Tensor>, filename: &str) -> Result<()> {
    let mut file = File::create(filename)?;

    for tensor in tensors {
        let shape_str = tensor
            .shape
            .iter()
            .map(|dim| dim.to_string())
            .collect::<Vec<String>>()
            .join(" ");
        writeln!(file, "{}", shape_str)?;

        match tensor.dtype {
            Dtype::Double => {
                let data_f64 = vec_raw_data_to_f64(&tensor.raw_data);

                let data_str = data_f64
                    .iter()
                    .map(|&val| val.to_string())
                    .collect::<Vec<String>>()
                    .join(" ");
                writeln!(file, "{}", data_str)?;
            }
            Dtype::I32 => {
                let data_i32 = vec_raw_data_to_i32(&tensor.raw_data);

                let data_str = data_i32
                    .iter()
                    .map(|&val| val.to_string())
                    .collect::<Vec<String>>()
                    .join(" ");
                writeln!(file, "{}", data_str)?;
            }
            Dtype::U32 => {
                let data_u32 = vec_raw_data_to_u32(&tensor.raw_data);

                let data_str = data_u32
                    .iter()
                    .map(|&val| val.to_string())
                    .collect::<Vec<String>>()
                    .join(" ");
                writeln!(file, "{}", data_str)?;
            }
            Dtype::Bool => {
                let data_bool = vec_raw_data_to_bool(&tensor.raw_data);

                let data_str = data_bool
                    .iter()
                    .map(|&val| val.to_string())
                    .collect::<Vec<String>>()
                    .join(" ");
                writeln!(file, "{}", data_str)?;
            }
        }
    }

    Ok(())
}

/// Prepares the .txt files used for tests
///
///  `prepare_test` takes the name of an onnx graph, then decomposes the graph into a `PrimGraph` and then executes the graph with the input of the test. Finally it writes the result in a .txt file in the tests repo.
///
/// # Arguments
///
/// * `input_name: &str` - name the the input ONNX graph
/// * `output_name: &str` - name the the output .txt file
/// * `input:  Vec<Tensor<f64>>` - input tensors for the test
///
pub fn prepare_test(input_name: &str, output_name: &str, input: Vec<Tensor>) {
    let prim_graph = decomp_onnx_to_primops(format!("tests/models/{input_name}").as_str());

    let result = execute_primgraph(&prim_graph, input);

    write_result(result, format!("tests/models/{output_name}").as_str())
        .expect("Failed to write ONNX file");
}
