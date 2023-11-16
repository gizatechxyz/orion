import os
from pathlib import Path


BASE_PATH = "./tests/nodes"


class ModFile:
    def __init__(self):
        """
        Initialize a ModFile object.

        This method creates a new file with a .cairo extension in the BASE_PATH directory.
        If the directory doesn't exist, it's created. The contents of the file are then read
        into the buffer attribute.
        """
        self.path = Path(f"{BASE_PATH}.cairo")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("r") as f:
            self.buffer = f.readlines()

    def update(self, name: str):
        """
        Update the .cairo file with a new module statement.

        Args:
            name (str): The name of the module to be added.

        This method checks if a module statement for the given name already exists in the buffer.
        If it doesn't, the new module statement is appended to the file.
        """
        statement = f"mod {name};"
        if any([line.startswith(statement) for line in self.buffer]):
            return

        with self.path.open("a") as f:
            f.write(f"{statement}\n")


class File:
    def __init__(self, path: str):
        """
        Initialize a File object.

        Args:
            path (str): The file path where the File object will operate.

        This method creates a new file at the specified path. If the file already exists, its
        contents are read into the buffer attribute.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.buffer = []

        if os.path.isfile(path):
            with self.path.open("r") as f:
                self.buffer = f.readlines()

    def dump(self):
        """
        Write the contents of the buffer to the file.

        This method writes each line in the buffer to the file, ensuring each line is
        properly terminated with a newline character.
        """
        with self.path.open("w") as f:
            f.writelines([f"{line}\n" for line in self.buffer])


class CairoTest(File):
    def __init__(self, file: str):
        super().__init__(os.path.join(BASE_PATH, file))

    @classmethod
    def base_template(cls, name: str, arg_cnt: int, refs: list[str], func_sig: str) -> list[str]:
        """
        Create a template for a Cairo test function which expects a tensor output.

        Args:
            name (str): Name of the test function.
            arg_cnt (int): Number of arguments for the function.
            refs (list[str]): List of references (modules) to be used in the function.
            func_sig (str): The function signature.

        Returns:
            list[str]: A list of strings that together form the template of a Cairo test function.

        This method generates a list of strings that form the template of a Cairo test function,
        including module imports, function definition, and assertions.
        """
        return [
            *[f"mod input_{i};" for i in range(arg_cnt)],
            *[ "mod output_0;"],
            *[ ""],
            *[ ""],
            *[f"use {ref};" for ref in refs],
            *[ ""],
            *[ "#[test]"],
            *[ "#[available_gas(2000000000)]"],
            *[f"fn test_{name}()"+" {"],
            *[f"    let input_{i} = input_{i}::input_{i}();" for i in range(arg_cnt)],
            *[ "    let z = output_0::output_0();"],
            *[ ""],
            *[f"    let y = {func_sig};"],
            *[ ""],
            *[ "    assert_eq(y, z);"],
            *[ "}"],
        ]

    @classmethod
    def sequence_template(cls, name: str, arg_cnt: int, refs: list[str], func_sig: str) -> list[str]:
        """
        Create a template for a Cairo test function which expects a tensor sequence.

        Args:
            name (str): Name of the test function.
            arg_cnt (int): Number of arguments for the function.
            refs (list[str]): List of references (modules) to be used in the function.
            func_sig (str): The function signature.

        Returns:
            list[str]: A list of strings that together form the template of a Cairo test function.

        This method generates a list of strings that form the template of a Cairo test function,
        including module imports, function definition, and assertions.
        """
        return [
            *[f"mod input_{i};" for i in range(arg_cnt)],
            *[ "mod output_0;"],
            *[ ""],
            *[ ""],
            *[f"use {ref};" for ref in refs],
            *[ ""],
            *[ "#[test]"],
            *[ "#[available_gas(2000000000)]"],
            *[f"fn test_{name}()"+" {"],
            *[f"    let input_{i} = input_{i}::input_{i}();" for i in range(arg_cnt)],
            *[ "    let z = output_0::output_0();"],
            *[ ""],
            *[f"    let y = {func_sig};"],
            *[ ""],
            *[ "    assert_seq_eq(y, z);"],
            *[ "}"],
        ]


class CairoData(File):
    def __init__(self, file: str):
        super().__init__(os.path.join(BASE_PATH, file))

    @classmethod
    def base_template(cls, func: str, dtype: str, refs: list[str], data: list[str], shape: tuple) -> list[str]:
        """
        Create a base template for data representation in Cairo.

        Args:
            func (str): The function name.
            dtype (str): The data type of the tensor.
            refs (list[str]): A list of module references.
            data (list[str]): The data to be included in the tensor.
            shape (tuple): The shape of the tensor.

        Returns:
            list[str]: A list of strings that together form the template of a data function in Cairo.

        This method generates a list of strings representing a function in Cairo for data handling,
        defining the shape and contents of a tensor.
        """
        template = [
            *[f"use {ref};" for ref in refs],
            *[ ""],
            *[f"fn {func}() -> Tensor<{dtype}>"+" {"],
            *[ "    let mut shape = ArrayTrait::<usize>::new();"],
            *[f"    shape.append({s});" for s in shape],
            *[ ""],
            *[ "    let mut data = ArrayTrait::new();"],
            *[f"    data.append({d});" for d in data],
            *[ "    TensorTrait::new(shape.span(), data.span())"],
            *[ "}"],
        ]

        return template

    @classmethod
    def sequence_template(cls, func: str, dtype: str, refs: list[str], data: list[list[str]], shape: list[tuple]) -> list[str]:
        """
        Create a template for handling tensor sequences in Cairo.

        Args:
            func (str): The function name.
            dtype (str): The data type of the tensor sequence.
            refs (list[str]): A list of module references.
            data (list[list[str]]): The data to be included in each tensor.
            shape (list[tuple]): The shapes of each tensor in the sequence.

        Returns:
            list[str]: A list of strings that together form the template of a sequence tensor function in Cairo.

        This method generates a list of strings representing a function in Cairo for handling a sequence
        of tensors, each with its own data and shape.
        """
        def expand_sequence_init(s: list[tuple], d: list[list[str]]) -> list[str]:
            snippet = []
            for i in range(len(s)):
                snippet += [
                    *[ "    let mut shape = ArrayTrait::<usize>::new();"],
                    *[f"    shape.append({s});" for s in s[i]],
                    *[ ""],
                    *[ "    let mut data = ArrayTrait::new();"],
                    *[f"    data.append({d});" for d in d[i]],
                    *[ ""],
                    *[ "    sequence.append(TensorTrait::new(shape.span(), data.span()));"],
                    *[ ""],
                ]

            return snippet

        template = [
            *[f"use {ref};" for ref in refs],
            *[ ""],
            *[f"fn {func}() -> Array<Tensor<{dtype}>>"+" {"],
            *[ "    let mut sequence = ArrayTrait::new();"],
            *[ ""],
            *expand_sequence_init(shape, data),
            *[ "    sequence"],
            *[ "}"],
        ]

        return template
