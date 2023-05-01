# Contributing Guidelines

New contributors are welcome to implement new ONNX Operators in Cairo 1.0! To see the full list of operators available click [here](https://github.com/onnx/onnx/blob/main/docs/Operators.md).

* Start by reading the [Engineering Design](https://onnxruntime.ai/docs/reference/high-level-design.html). More documentation can be found in the [repo docs folder](../../) and [on the repo wiki](https://github.com/microsoft/onnxruntime/wiki), as well as the user facing docs on the [ONNX Runtime website](https://onnxruntime.ai/docs).
* Read documentation about [Cairo](https://www.cairo-lang.org/resource-guide/) and the [context surrounding Cairo](https://perama-v.github.io/cairo/description/).

## 📝 Process details

Please search the [issue tracker](https://github.com/franalgaba/onnx-cairo/issues) for a similar idea first: there may already be an issue you can contribute to.

1. **Create Issue** To propose a new feature or API please start by filing a new issue in the [issue tracker](https://github.com/franalgaba/onnx-cairo/issues). Include as much detail as you have. It's fine if it's not a complete design: just a summary and rationale is a good starting point.
2. **Discussion** We'll keep the issue open for community discussion until it has been resolved or is deemed no longer relevant. Note that if an issue isn't a high priority or has many open questions then it might stay open for a long time.
3. **Owner Review** The ONNX Runtime team will review the proposal and either approve or close the issue based on whether it broadly aligns with the contribution guidelines.
4. **Implementation**

* A feature can be implemented by you, the ONNX Cairo Runtime team, or other community members. Code contributions are greatly appreciated: feel free to work on any reviewed feature you proposed, or choose one in the backlog and send us a PR. If you are new to the project and want to work on an existing issue, we recommend starting with issues that are tagged with “good first issue”. Please let us know in the issue comments if you are actively working on implementing a feature so we can ensure it's assigned to you.
* Unit tests: New code _must_ be accompanied by unit tests.
* Documentation and sample updates: If the PR affects any of the documentation or samples then include those updates in the same PR.
* Once a feature is complete and tested according to the contribution guidelines follow these steps:
  * Follow the [standard GitHub process to open a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests)
  * Add reviewers who have context from the earlier discussion. If you can't find a reviewer, add 'franalgaba'.
