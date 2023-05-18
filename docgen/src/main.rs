use std::fs;
use std::collections::HashMap;
use regex::Regex; 

fn main() {
    tensor_description();
    tensor_doc();
}

fn tensor_description() {
    // Read the core.cairo file
    let core_contents = fs
        ::read_to_string("../src/operators/tensor/core.cairo")
        .expect("Something went wrong reading the core.cairo file");

    // Extract the table from the core.cairo contents
    let start_index = core_contents.find("function").unwrap();
    let end_index = core_contents.find("trait TensorTrait<T>").unwrap();
    let table = core_contents[start_index..end_index].to_string();

    // Remove the /// from the table
    let table = table.replace("/// ", "");

    // Read the README.md file
    let mut readme_contents = fs
        ::read_to_string("../docs/apis/operators/tensor/README.md")
        .expect("Something went wrong reading the README.md file");

    // Find the location of the existing table in the README.md contents
    let readme_start_index = readme_contents.find("function").unwrap();
    let readme_end_index = readme_contents.find("### Arithmetic Operations").unwrap();

    // Replace the existing table in the README.md contents with the table from core.cairo
    readme_contents.replace_range(readme_start_index..readme_end_index - 1, &table);

    // Write the updated contents back to the README.md file
    fs::write("../docs/apis/operators/tensor/README.md", readme_contents).expect(
        "Something went wrong writing to the README.md file"
    );
}

fn tensor_doc() {
    // Read the core.cairo file
    let core_contents = fs
        ::read_to_string("../src/operators/tensor/core.cairo")
        .expect("Something went wrong reading the core.cairo file");

    // Split the core.cairo contents into lines
    let lines: Vec<&str> = core_contents.split('\n').collect();

    // Initialize a hashmap to store function documentation
    let mut function_docs: HashMap<String, String> = HashMap::new();

    // String to temporarily store the function name and its docs
    let mut function_name: String = String::new();
    let mut function_docs_temp: String = String::new();

    // Boolean to store if we are currently inside the TensorTrait block
    let mut in_tensor_trait_block = false;

    // Parse the lines to find function documentation
    for line in lines {
        if line.contains("trait TensorTrait<T> {") {
            in_tensor_trait_block = true;
            continue;
        }

        if line.contains("}") && in_tensor_trait_block {
            in_tensor_trait_block = false;
            continue;
        }

        if in_tensor_trait_block {
            let re = Regex::new(r"^.* fn ([a-zA-Z_][a-zA-Z_0-9]*)\(").unwrap();
            if let Some(cap) = re.captures(line) {
                function_name = cap[1].to_string();
            } else if line.starts_with("/// ") {
                function_docs_temp.push_str(&line.replace("/// ", ""));
                function_docs_temp.push_str("\n");
            } else if line == "}" {
                function_docs.insert(function_name.clone(), function_docs_temp.clone());
                function_docs_temp.clear();
            }
        }
    }

    // For each function, replace its documentation in the respective markdown file
    for (function_name, docs) in function_docs {
        let file_path = format!("../docs/apis/operators/tensor/{}.md", function_name);
        fs::write(&file_path, docs).expect(
            "Something went wrong writing to the function documentation file"
        );
    }
}