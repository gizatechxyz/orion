use onnx_decomp::utils::prepare_test;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        println!("Usage: {} <input_name> <output_name>", args[0]);
        return;
    }

    let input_name = &args[1];
    let output_name = &args[2];

    prepare_test(input_name, output_name)
}
