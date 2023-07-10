# Contribute

First of all, we sincerely appreciate your interest. It is precisely through contributions like yours that we can expand Orion and build a robust and transparent AI ecosystem

## 🚀 Getting Started

### Check the Issues

Before you start contributing, please check the [Issue Tracker](https://github.com/gizatechxyz/orion/issues) to see if there are any existing issues that match what you're intending to do. If the issue doesn't exist, please create it.

If you're creating a new issue, please provide a descriptive title and detailed description. If possible, include a code sample or an executable test case demonstrating the expected behavior that is not occurring.

### Fork and Clone the Repository

Once you've found an issue to work on, the next step is to fork the Orion repo and clone it to your local machine. This is necessary because you probably won't have push access to the main repo.

## ✍️ Making Changes

When you're ready to start coding, create a new branch on your cloned repo. It's important to use a separate branch for each issue you're working on. This keeps your changes separate in case you want to submit more than one contribution.

Please use meaningful names for your branches. For example, if you're working on a bug with the softmax function, you might name your branch `fix-softmax-bug`.

As you're making changes, make sure you follow the coding conventions used throughout the Orion project. Consistent code style makes it easier for others to read and understand your code.

### Implement new operators

Orion operators serve as the foundational components of machine learning models compliant with ONNX ops. You can follow this [step-by-step tutorial](../academy/tutorials/implement-new-operators-in-orion.md) to understand the process of implementing new operators within Orion.

## 🔥 Submitting a Pull Request

Once your changes are ready, commit them and push the branch to your forked repo on GitHub. Then you can open a pull request from your branch to the `develop` branch of the Orion repo.

When you submit the pull request, please provide a clear, detailed description of the changes you've made. If you're addressing a specific issue, make sure you reference it in the description.

Your pull request will be reviewed by the maintainers of the Orion project. They may ask for changes or clarification on certain points. Please address their comments and commit any required changes to the same branch on your repo.

## 🐎 Running Tests

Before you submit your pull request, you should run the test suite locally to make sure your changes haven't broken anything.

Additionally, when you push your changes, the built-in Continuous Integration (CI) will also run all the tests on the pushed code. You can see the result of these tests in the GitHub interface of your pull request. If the tests fail, you'll need to revise your code and push it again.

## 📜 Documentation

We strive to provide comprehensive, up-to-date documentation for Orion. If your changes require updates to the documentation, please include those in your pull request.&#x20;

If you implemented a new operator, please, run `scarb run docgen` to generate the documentation from docstring. [Read more](../academy/tutorials/implement-new-operators-in-orion.md#step-4-write-the-docstring) about docstrings in Orion Operators.

## 🎁 Getting Rewarded

Orion contributions are rewarded through [OnlyDust](https://app.onlydust.xyz/projects/32e92e68-13a5-4859-a122-69c0e738a8d1).\
It means that once the PR is merged you will be paid according to the time/difficulty of the PR.

## 🫶 Final Notes

Again, thank you for considering to contribute to Orion. Your contribution is invaluable to us. We hope this guide makes the contribution process clear and answers any questions you might have. If not, feel free to ask on the [Discord](https://discord.gg/yqWB57XNYg) or on [GitHub](https://github.com/gizatechxyz/orion/issues).
