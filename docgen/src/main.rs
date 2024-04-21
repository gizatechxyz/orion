use regex::Regex;
use std::fs;
use std::path::Path;

fn main() {
    // TENSOR DOC
    let trait_path = "src/operators/tensor/core.cairo";
    let doc_path = "docs/framework/operators/tensor";
    let label = "tensor";
    let trait_name = "TensorTrait";
    doc_trait(trait_path, doc_path, label);
    doc_functions(trait_path, doc_path, trait_name, label);

    // NN DOC
    let trait_path = "src/operators/nn/core.cairo";
    let doc_path = "docs/framework/operators/neural-network";
    let label = "nn";
    let trait_name = "NNTrait";
    doc_trait(trait_path, doc_path, label);
    doc_functions(trait_path, doc_path, trait_name, label);

    // SEQUENCE DOC
    let trait_path = "src/operators/sequence/core.cairo";
    let doc_path = "docs/framework/operators/sequence";
    let label = "sequence";
    let trait_name = "SequenceTrait";
    doc_trait(trait_path, doc_path, label);
    doc_functions(trait_path, doc_path, trait_name, label);

    // FIXED POINT DOC
    let trait_path = "src/numbers/fixed_point/core.cairo";
    let doc_path = "docs/framework/numbers/fixed-point";
    let label = "fp";
    let trait_name = "FixedTrait";
    doc_trait(trait_path, doc_path, label);
    doc_functions(trait_path, doc_path, trait_name, label);

    // COMPLEX NUMBER DOC
    let trait_path = "src/numbers/complex_number/complex_trait.cairo";
    let doc_path = "docs/framework/numbers/complex-number";
    let label = "complex";
    let trait_name: &str = "ComplexTrait";
    doc_trait(trait_path, doc_path, label);
    doc_functions(trait_path, doc_path, trait_name, label);

    // TREE ENSEMBLE CLASSIFIER DOC
    let trait_path = "src/operators/ml/tree_ensemble/tree_ensemble_classifier.cairo";
    let doc_path = "docs/framework/operators/machine-learning/tree-ensemble-classifier";
    let label = "tree_ensemble_classifier";
    let trait_name: &str = "TreeEnsembleClassifierTrait";
    doc_trait(trait_path, doc_path, label);
    doc_functions(trait_path, doc_path, trait_name, label);

    // TREE ENSEMBLE REGRESSOR DOC
    let trait_path = "src/operators/ml/tree_ensemble/tree_ensemble_regressor.cairo";
    let doc_path = "docs/framework/operators/machine-learning/tree-ensemble-regressor";
    let label = "tree_ensemble_regressor";
    let trait_name: &str = "TreeEnsembleRegressorTrait";
    doc_trait(trait_path, doc_path, label);
    doc_functions(trait_path, doc_path, trait_name, label);

    // TREE ENSEMBLE DOC
    let trait_path = "src/operators/ml/tree_ensemble/tree_ensemble.cairo";
    let doc_path = "docs/framework/operators/machine-learning/tree-ensemble";
    let label = "tree_ensemble";
    let trait_name: &str = "TreeEnsembleTrait";
    doc_trait(trait_path, doc_path, label);
    doc_functions(trait_path, doc_path, trait_name, label);

    // LINEAR REGRESSOR DOC
    let trait_path = "src/operators/ml/linear/linear_regressor.cairo";
    let doc_path = "docs/framework/operators/machine-learning/linear-regressor";
    let label = "linear_regressor";
    let trait_name: &str = "LinearRegressorTrait";
    doc_trait(trait_path, doc_path, label);
    doc_functions(trait_path, doc_path, trait_name, label);

    // LINEAR CLASSIFIER DOC
    let trait_path = "src/operators/ml/linear/linear_classifier.cairo";
    let doc_path = "docs/framework/operators/machine-learning/linear-classifier";
    let label = "linear_classifier";
    let trait_name: &str = "LinearClassifierTrait";
    doc_trait(trait_path, doc_path, label);
    doc_functions(trait_path, doc_path, trait_name, label);

    // SVM REGRESSOR DOC
    let trait_path = "src/operators/ml/svm/svm_regressor.cairo";
    let doc_path = "docs/framework/operators/machine-learning/svm-regressor";
    let label = "svm_regressor";
    let trait_name: &str = "SVMRegressorTrait";
    doc_trait(trait_path, doc_path, label);
    doc_functions(trait_path, doc_path, trait_name, label);

    // SVM CLASSIFIER DOC
    let trait_path = "src/operators/ml/svm/svm_classifier.cairo";
    let doc_path = "docs/framework/operators/machine-learning/svm-classifier";
    let label = "svm_classifier";
    let trait_name: &str = "SVMClassifierTrait";
    doc_trait(trait_path, doc_path, label);
    doc_functions(trait_path, doc_path, trait_name, label);

    // NORMALIZER DOC
    let trait_path = "src/operators/ml/normalizer/normalizer.cairo";
    let doc_path = "docs/framework/operators/machine-learning/normalizer";
    let label = "normalizer";
    let trait_name: &str = "NormalizerTrait";
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
    let trait_re = Regex::new(&format!(
        r"(?s)trait\s+{}\s*(<[\w\s,]*>)?\s*\{{.*?\n\s*\}}",
        trait_name
    ))
    .unwrap();

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
            .map(|line| {
                line.trim_start().strip_prefix("/// ").unwrap_or(
                    line.trim_start()
                        .strip_prefix("///")
                        .unwrap_or(line.trim_start()),
                )
            })
            .collect::<Vec<_>>()
            .join("\n");

        // Write or replace the transformed comment into the markdown file
        fs::write(markdown_filename, transformed_comment).expect("Unable to write file");
    }
}
