use std::fs;
use std::path::Path;
use regex::Regex;

fn main() {
    doc_trait();
    doc_functions();
}

fn doc_trait() {
    // Open and read core.cairo file
    let path = Path::new("../src/operators/tensor/core.cairo");
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
            "[`tensor.{}`](tensor.{}.md)",
            &cap[1],
            &cap[1].replace("_", "\\_")
        );
        let func_desc = &cap[2];
        table += &format!("| {} | {} |\n", func_name, func_desc);
    }

    // Open the README.md file
    let readme_path = Path::new("../docs/apis/operators/tensor/README.md");
    let readme = fs::read_to_string(&readme_path).expect("Could not read the file");

    // Use regex to replace the table, including the "| fun" line and two empty lines before and after
    let re_table = Regex::new(r"(?ms)\n\n\| fun.*?\n\n").unwrap();
    let new_readme = re_table.replace(&readme, &("\n\n".to_owned() + &table + "\n\n"));

    // Write the updated contents back to README.md
    fs::write(&readme_path, &*new_readme).expect("Could not write the file");
}

fn doc_functions() {
    // Step 1 and 2: Go to the file and read it into a string
    let filepath = "../src/operators/tensor/core.cairo";
    let contents = fs::read_to_string(filepath).expect("Something went wrong reading the file");

    // Step 3: Find the TensorTrait block
    let trait_re = Regex::new(r"(?s)trait\s+TensorTrait<T>\s*\{.*?\n\s*\}").unwrap();
    let trait_match = trait_re.captures(&contents).unwrap();
    let trait_block = trait_match.get(0).unwrap().as_str();

    // Step 4: Iterate over each function
    let func_re = Regex::new(r"(?s)(///.*?\n)\s*fn (\w+)\((.*?)\) -> (.*?);").unwrap();
    for func_match in func_re.captures_iter(trait_block) {
        let func_name = func_match.get(2).unwrap().as_str();
        let doc_comment = func_match.get(1).unwrap().as_str();

        // Step 5 and 6: Go to the appropriate markdown file and write the transformed doc comment
        let markdown_filename = format!("../docs/apis/operators/tensor/tensor.{}.md", func_name);

        let transformed_comment = doc_comment
            .lines()
            .map(|line| line.trim_start().trim_start_matches("///").trim())
            .collect::<Vec<_>>()
            .join("\n");

        // Step 7: Write or replace the transformed comment into the markdown file
        fs::write(markdown_filename, transformed_comment).expect("Unable to write file");
    }
}