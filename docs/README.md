# ONNX-Cairo Runtime

ONNX-Cairo Runtime is a Cairo library that provides two high-level features:

* Tensor computation (like Numpy) in Cairo 1.0.
* Verifiable Machine Learning models using STARKS.

### 🤔 What is ONNX Runtime inference?

ONNX Runtime is an open-source, high-performance inference engine for machine learning models in the Open Neural Network Exchange (ONNX) format. ONNX is an interoperable format that allows deep learning models to be represented, shared, and executed across different AI frameworks and platforms.

ONNX Runtime inference can enable faster user experiences and lower costs, supporting models from deep learning frameworks such as PyTorch and TensorFlow/Keras as well as classical machine learning libraries such as scikit-learn, LightGBM, XGBoost, etc. ONNX Runtime is compatible with various hardware, drivers, and operating systems, and provides optimal performance by leveraging hardware accelerators where applicable alongside graph optimizations and transforms. [Learn more →](https://www.onnxruntime.ai/docs/#onnx-runtime-for-inferencing)

This library proposes a new ONNX runtime built with [Cairo](https://www.cairo-lang.org/). The purpose is to provide a runtime implementation for verifiable ML model inferences using [STARKs](https://starkware.co/stark/).

### 🌱 Where to start?

<table data-view="cards"><thead><tr><th align="center"></th><th></th><th></th><th data-hidden data-card-target data-type="content-ref"></th></tr></thead><tbody><tr><td align="center"></td><td>🧱 <strong>APIs</strong></td><td>Three APIs that will help you to build your Validity ML models.</td><td><a href="apis/get-started.md">get-started.md</a></td></tr><tr><td align="center"></td><td>🧩 <strong>Algorithms</strong></td><td>Discover all algorithms built by the community, or build your own. </td><td><a href="community/algorithms.md">algorithms.md</a></td></tr><tr><td align="center"></td><td>📖 <strong>Tutorials</strong></td><td>Try out the awesome guides and tutorials created by the community.</td><td><a href="resources/tutorials.md">tutorials.md</a></td></tr></tbody></table>

### ✨ What's new?

For a detailed list of changes, please refer to the [CHANGELOG](https://github.com/franalgaba/onnx-cairo/blob/main/docs/CHANGELOG.md) file.

### 💖 Join the community!

Join the community and help build a safer and transparent AI in our [Discord](https://discord.gg/Kt24CsMb5k)!

### ✍️ Authors & contributors

For a full list of all authors and contributors, see [the contributors page](https://github.com/franalgaba/onnx-cairo/graphs/contributors).

### License

This project is licensed under the **MIT license**.

See [LICENSE](https://github.com/franalgaba/onnx-cairo/blob/main/LICENSE/README.md) for more information.
