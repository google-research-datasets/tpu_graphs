load(
    "//third_party/tensorflow:tensorflow.bzl",
    "tf_gen_op_wrapper_py",
    "tf_kernel_library",
)

package(
    default_applicable_licenses = ["//third_party/py/tpu_graphs:license"],
    default_visibility = ["//third_party/py/tpu_graphs:internal"],
    licenses = ["notice"],
)

filegroup(
    name = "testdata",
    srcs = [
        "testdata/module_tuning_data_layout.pb",
        "testdata/op_tuning_data.pb",
    ],
)

tf_kernel_library(
    name = "encode_hlo_tuning_data.so",
    srcs = ["encode_hlo_tuning_data.cc"],
    deps = [
        "//third_party/py/tpu_graphs/process_data/xla:featurizers",
        "//third_party/py/tpu_graphs/process_data/xla:hlo_encoder",
        "//third_party/py/tpu_graphs/process_data/xla:hlo_opcode",
        "//third_party/py/tpu_graphs/process_data/xla:tuning_data_iterator",
        "//third_party/py/tpu_graphs/proto:tuning_cc_proto",
        "//third_party/tensorflow/compiler/xla:statusor",
        "//third_party/tensorflow/compiler/xla/hlo/ir:hlo",
        "//third_party/tensorflow/core:framework",
        "//third_party/tensorflow/core:framework_lite",
        "//third_party/tensorflow/core:framework_types_hdr",
        "//third_party/tensorflow/core:lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "encode_hlo_tuning_data",
    out = "encode_hlo_tuning_data.py",
    op_allowlist = [
        "EncodeHloTuningData",
    ],
    deps = [":encode_hlo_tuning_data.so"],
)

py_test(
    name = "encode_hlo_tuning_data_test",
    srcs = ["encode_hlo_tuning_data_test.py"],
    data = [
        "testdata/op_tuning_data.pb",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":encode_hlo_tuning_data",
        ":encode_hlo_tuning_data.so",
        "//pyglib:resources",
        "//testing/pybase",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

tf_kernel_library(
    name = "encode_hlo_module_data.so",
    srcs = ["encode_hlo_module_data.cc"],
    deps = [
        "//third_party/py/tpu_graphs/process_data/xla:hlo_encoder",
        "//third_party/py/tpu_graphs/process_data/xla:hlo_opcode",
        "//third_party/py/tpu_graphs/process_data/xla:tuning_data_iterator",
        "//third_party/py/tpu_graphs/proto:tuning_cc_proto",
        "//third_party/tensorflow/compiler/xla:statusor",
        "//third_party/tensorflow/compiler/xla/hlo/ir:hlo",
        "//third_party/tensorflow/compiler/xla/service:hlo_proto_cc",
        "//third_party/tensorflow/core:framework",
        "//third_party/tensorflow/core:lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "encode_hlo_module_data",
    out = "encode_hlo_module_data.py",
    op_allowlist = [
        "EncodeHloModuleData",
    ],
    deps = [":encode_hlo_module_data.so"],
)

py_test(
    name = "encode_hlo_module_data_test",
    srcs = ["encode_hlo_module_data_test.py"],
    data = [
        "testdata/module_tuning_data_layout.pb",
    ],
    deps = [
        ":encode_hlo_module_data",
        ":encode_hlo_module_data.so",
        "//pyglib:resources",
        "//testing/pybase",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)

tf_kernel_library(
    name = "encode_hlo_config_data.so",
    srcs = ["encode_hlo_config_data.cc"],
    deps = [
        "//third_party/py/tpu_graphs/process_data/xla:hlo_opcode",
        "//third_party/py/tpu_graphs/process_data/xla:tuning_data_iterator",
        "//third_party/tensorflow/compiler/xla:statusor",
        "//third_party/tensorflow/compiler/xla/hlo/ir:hlo",
        "//third_party/tensorflow/core:framework",
        "//third_party/tensorflow/core:lib",
    ],
)

tf_gen_op_wrapper_py(
    name = "encode_hlo_config_data",
    out = "encode_hlo_config_data.py",
    op_allowlist = [
        "EncodeHloConfigData",
    ],
    deps = [":encode_hlo_config_data.so"],
)

py_test(
    name = "encode_hlo_config_data_test",
    srcs = ["encode_hlo_config_data_test.py"],
    data = [
        "testdata/module_tuning_data_layout.pb",
    ],
    deps = [
        ":encode_hlo_config_data",
        ":encode_hlo_config_data.so",
        "//pyglib:resources",
        "//pyglib:retry",
        "//testing/pybase",
        "//third_party/py/tensorflow:tensorflow_no_contrib",
    ],
)
