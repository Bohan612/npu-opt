import os
import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

config.name = "npu-opt"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.test_source_root = os.path.join(os.path.dirname(__file__), "npu-opt")
config.test_exec_root = os.path.join(config.npu_obj_root, "test", "npu-opt")

llvm_config.with_environment("PATH", config.npu_tools_dir, append_path=True)
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tools = [
    ToolSubst("npu-opt", command=os.path.join(config.npu_tools_dir, "npu-opt")),
    ToolSubst("FileCheck", command=os.path.join(config.llvm_tools_dir, "FileCheck")),
]
llvm_config.add_tool_substitutions(tools)
