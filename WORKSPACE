# Set up dependencies for Bazel.

workspace(name = "tpu_graphs")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Must come before TensorFlow because TensorFlow has out-dated pybind.
http_archive(
    name = "rules_python",
    sha256 = "8c8fe44ef0a9afc256d1e75ad5f448bb59b81aba149b8958f02f7b3a98f5d9b4",
    strip_prefix = "rules_python-0.13.0",
    url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.13.0.tar.gz",
)

load("@rules_python//python:repositories.bzl", "python_register_toolchains")

python_register_toolchains(
    name = "python3_10",
    python_version = "3.10",
)

load("@python3_10//:defs.bzl", "interpreter")

load("@rules_python//python:pip.bzl", "pip_parse")

pip_parse(
    python_interpreter_target = interpreter,
)

http_archive(
  name = "pybind11_bazel",
  strip_prefix = "pybind11_bazel-9a24c33cbdc510fa60ab7f5ffb7d80ab89272799",
  urls = ["https://github.com/pybind/pybind11_bazel/archive/9a24c33cbdc510fa60ab7f5ffb7d80ab89272799.zip"],
)
http_archive(
  name = "pybind11",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-2.10.0",
  urls = ["https://github.com/pybind/pybind11/archive/refs/tags/v2.10.0.zip"],
)
load("@pybind11_bazel//:python_configure.bzl", "python_configure")

python_configure(
    name = "local_config_python",
    # python_interpreter_target = interpreter,
)

# To update TensorFlow to a new revision,
# a) update URL and strip_prefix to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the sha256 with the result.

http_archive(
    name = "org_tensorflow",
    sha256 = "622a92e22e6f3f4300ea43b3025a0b6122f1cc0e2d9233235e4c628c331a94a3",
    strip_prefix = "tensorflow-2.10.1",
    urls = ["https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.10.1.tar.gz"],
)

# This can be used to build against a local version of Tensorflow.

# local_repository(
#     name = "org_tensorflow",
#     path = "tensorflow/",
# )

# Initialize TensorFlow's external dependencies.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")
tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")
tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")
tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")
tf_workspace0()

# This com_google_protobuf repository is required for proto_library rule.
# It provides the protocol compiler binary (i.e., protoc).
http_archive(
    name = "com_google_protobuf",
    strip_prefix = "protobuf-master",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/master.zip"],
)

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()
