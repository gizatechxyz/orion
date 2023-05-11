# Get Started

In this section, we will guide you to start using our APIs successfully. We will help you install Cairo 1.0 and add ONNX Cairo dependency in your project.

## 📦 Installations

<details>

<summary>Install Cairo 1.0</summary>

#### Step 1: Install Cairo 1.0

There are different ways to install Cairo 1.0. Use the one that suits you best: [Cairo 1.0 installer](https://cairo-book.github.io/ch01-01-installation.html).

#### Step 2: Setup Language Server

Install the Cairo 1 **VS Code Extension** for proper syntax highlighting and code navigation. Just follow the steps indicated [here](https://github.com/starkware-libs/cairo/blob/main/vscode-cairo/README.md).

</details>

<details>

<summary>Install the Cairo package manager Scarb</summary>

#### Step 1: Install Scarb

Follow the installation guide on the [Scarb's Website](https://docs.swmansion.com/scarb/download).

#### Step 2: Create a new Scarb project

Follow the instructions [here](https://docs.swmansion.com/scarb/docs/guides/creating-a-new-package) to start a new Scarb project.

</details>

## ⚙️ Add `onnx_cairo` dependency in your project

If your `Scarb.toml` doesn't already have a `[dependencies]` section, add it, then list the package name and the URL to its Git repository.

{% code title="Scarb.toml" %}
```toml
[dependencies]
onnx_cairo = { git = "https://github.com/gizatechxyz/onnx-cairo" }
```
{% endcode %}

Now, run `scarb build`, and Scarb will fetch `onnx_cairo` dependency and all its dependencies. Then it will compile your package with all of these packages included:

```sh
scarb build
```

You can now use the `onnx_cairo` in your files:

```rust
use onnx_cairo::operators::nn::nn_i32::NN;

fn relu_example() -> Tensor<u32> {
    let tensor = u32_tensor_2x2_helper();
    return NN::relu(@tensor);
}
```

## 🔭 Discover the ONNX-Cairo APIs

<table data-view="cards"><thead><tr><th></th><th></th><th></th></tr></thead><tbody><tr><td></td><td>⚙️ <strong>Operators</strong></td><td>A set of standardized math functions that are used in the computation of neural network models.</td></tr><tr><td></td><td>🔢 <strong>Numbers</strong></td><td>A full implementation of Signed Integer and Fixed Point Q5.26 in Cairo 1.0.</td></tr><tr><td></td><td>🚀 <strong>Performance</strong></td><td>A set of functions to increase the performance of your model.</td></tr></tbody></table>
