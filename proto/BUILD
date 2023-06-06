load("//tools/build_defs/proto/cpp:cc_proto_library.bzl", "cc_proto_library")

package(
    default_applicable_licenses = ["//third_party/py/tpu_graphs:license"],
    default_visibility = ["//visibility:public"],
    licenses = ["notice"],
)

proto_library(
    name = "tuning_proto",
    srcs = ["tuning.proto"],
    cc_api_version = 2,
    compatible_with = [
        "//buildenv/target:libtpu",
        "//buildenv/target:non_prod",
    ],
    deps = [
        "//third_party/tensorflow/compiler/xla:xla_data_proto",
        "//third_party/tensorflow/compiler/xla/service:hlo_proto",
    ],
)

cc_proto_library(
    name = "tuning_cc_proto",
    compatible_with = ["//buildenv/target:non_prod"],
    deps = [":tuning_proto"],
)

py_proto_library(
    name = "tuning_py_pb2",
    api_version = 2,
    deps = [":tuning_proto"],
)
