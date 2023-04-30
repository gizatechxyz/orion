## Getting Started

In this section you can find all the needed resources to setup the repository and start working with it.

### Installing dependencies

#### Step 1: Install Cairo 1.0

If you are on an x86 Linux system and able to use the release binary,
you can download Cairo here https://github.com/starkware-libs/cairo/releases.

For everyone, else, we recommend compiling Cairo from source like so:

```bash
# Install Rust
$ curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Update Rust
$ rustup override set stable && rustup update

# Clone the Cairo compiler in $HOME/Bin
$ mkdir ~/Bin && cd ~/Bin && git clone https://github.com/starkware-libs/cairo.git && cd cairo

# Generate release binaries
$ cargo build --all --release
```

**NOTE: Keeping Cairo up to date**

Now that your Cairo compiler is in a cloned repository, all you will need to do
is pull the latest changes and rebuild as follows:

```bash
$ cd ~/Bin/cairo && git fetch && git pull && cargo build --all --release
```

#### Step 2: Add Cairo 1.0 executables to your path

```bash
export PATH="$HOME/Bin/cairo/target/release:$PATH"
```

**NOTE: If installing from a Linux binary, adapt the destination path accordingly.**

This will make available several binaries. The one we use is called `cairo-test`.

#### Step 3: Install the Cairo package manager Scarb

Follow the installation guide in [Scarb's Website](https://docs.swmansion.com/scarb/download).

#### Step 4: Setup Language Server

##### VS Code Extension

- Install the Cairo 1 extension for proper syntax highlighting and code navigation.
Just follow the steps indicated [here](https://github.com/starkware-libs/cairo/blob/main/vscode-cairo/README.md).
