# Get Started

In this section, we will guide you to start using Orion successfully. We will help you install Cairo 1.0 and add Orion dependency in your project.

{% hint style="info" %}
Orion supports <mark style="color:orange;">**Cairo and Scarb v2.4.0**</mark>
{% endhint %}

## 📦 Installations

<details>

<summary>Install Cairo</summary>

**Step 1: Install Cairo**

There are different ways to install Cairo. Use the one that suits you best: [Cairo installer.](https://book.cairo-lang.org/ch01-01-installation.html)

**Step 2: Setup Language Server**

Install the Cairo 1 **VS Code Extension** for proper syntax highlighting and code navigation. Just follow the steps indicated [here](https://github.com/starkware-libs/cairo/blob/main/vscode-cairo/README.md).

</details>

<details>

<summary>Install the Cairo package manager Scarb</summary>

**Step 1: Install Scarb**

Follow the installation guide on the [Scarb's Website](https://docs.swmansion.com/scarb/download).

**Step 2: Create a new Scarb project**

Follow the instructions [here](https://docs.swmansion.com/scarb/docs/guides/creating-a-new-package) to start a new Scarb project.

</details>

## ⚙️ Add `orion` dependency in your project

If your `Scarb.toml` doesn't already have a `[dependencies]` section, add it, then list the package name and the URL to its Git repository.

{% code title="Scarb.toml" %}
```toml
[dependencies]
orion = { git = "https://github.com/gizatechxyz/onnx-cairo" }
```
{% endcode %}

Now, run `scarb build`, and Scarb will fetch `orion` dependency and all its dependencies. Then it will compile your package with all of these packages included:

```sh
scarb build
```

You can now use the `orion` in your files:

```rust
use core::array::{ArrayTrait, SpanTrait};

use orion::operators::tensor::{TensorTrait, Tensor, I32Tensor};
use orion::operators::nn::{NNTrait, I32NN};
use orion::numbers::signed_integer::i32::{i32, IntegerTrait};

fn relu_example() -> Tensor<i32> {
    let tensor = TensorTrait::<i32>::new(
        shape: array![2, 2].span(),
        data: array![
            IntegerTrait::new(1, false),
            IntegerTrait::new(2, false),
            IntegerTrait::new(1, true),
            IntegerTrait::new(2, true),
        ]
            .span(),
    );

    return NNTrait::relu(@tensor);
}
```

## 🔭 Discover the Orion APIs

<table data-view="cards"><thead><tr><th></th><th></th><th></th><th data-hidden data-card-target data-type="content-ref"></th></tr></thead><tbody><tr><td></td><td>⚙️ <strong>Operators</strong></td><td>A set of standardized math functions that are used in the computation of neural network models.</td><td><a href="operators/">operators</a></td></tr><tr><td></td><td>🔢 <strong>Numbers</strong></td><td>A full implementation of Signed Integer and Fixed Point in Cairo.</td><td><a href="numbers/">numbers</a></td></tr></tbody></table>
