use std::fs;
use std::path::Path;
use regex::Regex;

fn main() {
    // TENSOR DOC
    let trait_path = "src/operators/tensor/core.cairo";
    let doc_path = "docs/apis/operators/tensor";
    let label = "tensor";
    let trait_name = "TensorTrait";
    doc_trait(trait_path, doc_path, label);
    doc_functions(trait_path, doc_path, trait_name, label);

    // NN DOC
    let trait_path = "src/operators/nn/core.cairo";
    let doc_path = "docs/apis/operators/neural-network";
    let label = "nn";
    let trait_name = "NNTrait";
    doc_trait(trait_path, doc_path, label);
    doc_functions(trait_path, doc_path, trait_name, label);

    // FIXED POINT DOC
    let trait_path = "src/numbers/fixed_point/core.cairo";
    let doc_path = "docs/apis/numbers/fixed-point";
    let label = "fp";
    let trait_name = "FixedTrait";
    doc_trait(trait_path, doc_path, label);
    doc_functions(trait_path, doc_path, trait_name, label);

    // SIGNED INTEGER DOC
    let trait_path = "src/numbers/signed_integer/integer_trait.cairo";
    let doc_path = "docs/apis/numbers/signed-integer";
    let label = "int";
    let trait_name: &str = "IntegerTrait";
    doc_trait(trait_path, doc_path, label);
    doc_functions(trait_path, doc_path, trait_name, label);

    // PERFORMANCE DOC
    let trait_path = "src/performance/core.cairo";
    let doc_path = "docs/apis/performance";
    let label = "performance";
    let trait_name = "PerfomanceTrait";
    doc_trait(trait_path, doc_path, label);
    doc_functions(trait_path, doc_path, trait_name, label);
}

fn doc_trait(trait_path: &str, doc_path: &str, label: &str) {
    // Open and read core.cairo file
    let path_str = format!("../{}", trait_path);
    let path = Path::new(&path_str);
    let contents = fs::read_to_string(&path).expect("Could not read the file");

    // Create a regular expression to match the comment lines
    let re = Regex::new(r#"/// (\w+) - (.*)"#).unwrap();

    // Initialize an empty string to store our new formatted table
    let mut table = String::from("| function | description |\n| --- | --- |\n");

    // Go through the file and look for comments with our specific format
    for cap in re.captures_iter(&contents) {
        // Check if the function is the Trait definition and skip it
        if &cap[1] == "Trait" {
            continue;
        }

        // Add the function name and description to our table
        let func_name = format!(
            "[`{}.{}`]({}.{}.md)",
            label,
            &cap[1],
            label,
            &cap[1].replace('_', r"\_")
        );
        let func_desc = &cap[2];
        table += &format!("| {} | {} |\n", func_name, func_desc);
    }

    // Open the README.md file
    let readme_path_str = format!("../{}/README.md", doc_path);
    let readme_path = Path::new(&readme_path_str);
    let readme = fs::read_to_string(&readme_path).expect("Could not read the file");

    // Use regex to replace the table
    let re_table = Regex::new(r"(?ms)\n\n\| fun.*?(\n[^|]|\z)").unwrap();
    let new_readme = re_table.replace(&readme, &("\n\n".to_owned() + &table + "\n"));

    // Write the updated contents back to README.md
    fs::write(&readme_path, &*new_readme).expect("Could not write the file");
}

fn doc_functions(trait_path: &str, doc_path: &str, trait_name: &str, label: &str) {
    let filepath_str = format!("../{}", trait_path);
    let filepath = Path::new(&filepath_str);
    let contents = fs::read_to_string(filepath).expect("Something went wrong reading the file");

    // Find the trait block
    let trait_re = Regex::new(
        &format!(r"(?s)trait\s+{}\s*(<[\w\s,]*>)?\s*\{{.*?\n\s*\}}", trait_name)
    ).unwrap();

    let trait_match = trait_re.captures(&contents).unwrap();
    let trait_block = trait_match.get(0).unwrap().as_str();

    // Iterate over each function
    let func_re = Regex::new(r"(?s)(///.*?\n)\s*fn (\w+)\((.*?)\) -> (.*?);").unwrap();
    for func_match in func_re.captures_iter(trait_block) {
        let func_name = func_match.get(2).unwrap().as_str();
        let doc_comment = func_match.get(1).unwrap().as_str();

        // Go to the appropriate markdown file and write the transformed doc comment
        let markdown_filename = format!("../{}/{}.{}.md", doc_path, label, func_name);

        let transformed_comment = doc_comment
            .lines()
            .map(|line| line.trim_start().trim_start_matches("///").trim())
            .collect::<Vec<_>>()
            .join("\n");

        // Write or replace the transformed comment into the markdown file
        fs::write(markdown_filename, transformed_comment).expect("Unable to write file");
    }
}
