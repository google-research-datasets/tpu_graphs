package(
    default_applicable_licenses = ["//third_party/py/tpu_graphs:license"],
    default_visibility = ["//third_party/py/tpu_graphs:internal"],
    licenses = ["notice"],
)

cc_library(
    name = "hlo_opcode",
    hdrs = ["hlo_opcode.h"],
    compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/strings",
    ],
)

cc_library(
    name = "featurizers",
    hdrs = ["featurizers.h"],
    compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        ":hlo_opcode",
        "//third_party/absl/memory",
        "//third_party/absl/strings",
        "//third_party/absl/strings:str_format",
        "//third_party/py/tpu_graphs/proto:tuning_cc_proto",
        "//third_party/tensorflow/compiler/xla:types",
        "//third_party/tensorflow/compiler/xla:xla_data_proto_cc",
        "//third_party/tensorflow/compiler/xla/service:hlo_proto_cc",
        "//util/gtl:array",
    ],
)

cc_library(
    name = "hlo_encoder",
    srcs = ["hlo_encoder.cc"],
    hdrs = ["hlo_encoder.h"],
    compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        ":featurizers",
        ":hlo_opcode",
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/strings",
        "//third_party/eigen3",
        "//third_party/py/tpu_graphs/proto:tuning_cc_proto",
        "//third_party/tensorflow/compiler/xla:status",
        "//third_party/tensorflow/compiler/xla/hlo/ir:hlo",
        "//third_party/tensorflow/compiler/xla/service:hlo_proto_cc",
        "//third_party/tensorflow/core:framework",
        "//third_party/tensorflow/core:framework_lite",
    ],
)

cc_test(
    name = "hlo_encoder_test",
    srcs = ["hlo_encoder_test.cc"],
    deps = [
        ":hlo_encoder",
        "//testing/base/public:gunit_main",
        "//third_party/tensorflow/compiler/xla/tests:hlo_test_base",
        "//third_party/tensorflow/core:framework",
        "//third_party/tensorflow/core:framework_lite",
        "//third_party/tensorflow/core:protos_all_cc",
        "//third_party/tensorflow/tsl/lib/core:status_test_util",
    ],
)

cc_test(
    name = "hlo_opcode_test",
    srcs = ["hlo_opcode_test.cc"],
    deps = [
        ":hlo_opcode",
        "//testing/base/public:gunit_main",
    ],
)

cc_test(
    name = "featurizers_test",
    srcs = ["featurizers_test.cc"],
    deps = [
        ":featurizers",
        ":hlo_opcode",
        "//testing/base/public:gunit_main",
        "//third_party/absl/strings",
        "//third_party/tensorflow/compiler/xla/hlo/ir:hlo",
        "//third_party/tensorflow/compiler/xla/tests:hlo_test_base",
        "//third_party/tensorflow/core:test",
    ],
)

cc_library(
    name = "tuning_data_iterator",
    srcs = ["tuning_data_iterator.cc"],
    hdrs = ["tuning_data_iterator.h"],
    compatible_with = ["//buildenv/target:non_prod"],
    deps = [
        "//third_party/absl/container:flat_hash_map",
        "//third_party/absl/log:check",
        "//third_party/absl/random",
        "//third_party/absl/strings",
        "//third_party/absl/time",
        "//third_party/py/tpu_graphs/proto:tuning_cc_proto",
        "//third_party/tensorflow/compiler/xla:status",
        "//third_party/tensorflow/compiler/xla:statusor",
        "//third_party/tensorflow/compiler/xla:xla_data_proto_cc",
        "//third_party/tensorflow/compiler/xla/service:hlo_proto_cc",
        "//third_party/tensorflow/core:lib",
        "//third_party/tensorflow/core/platform:errors",
        "//third_party/tensorflow/core/platform:tstring",
    ],
)

cc_test(
    name = "tuning_data_iterator_test",
    srcs = ["tuning_data_iterator_test.cc"],
    data = [
        "//third_party/py/tpu_graphs/process_data/tensorflow_ops:testdata",
    ],
    deps = [
        ":tuning_data_iterator",
        "//testing/base/public:gunit_main",
        "//third_party/py/tpu_graphs/proto:tuning_cc_proto",
        "//third_party/tensorflow/compiler/xla:xla_data_proto_cc",
        "//third_party/tensorflow/compiler/xla/service:hlo_proto_cc",
        "//third_party/tensorflow/core:lib",
        "//third_party/tensorflow/core:test",
        "//third_party/tensorflow/tsl/platform",
    ],
)
