load(
    "//devtools/python/blaze:pytype.bzl",
    "pytype_strict_binary",
    "pytype_strict_library",
    pytype_strict_test = "pytype_strict_contrib_test",
)

package(
    default_applicable_licenses = ["//third_party/py/tpu_graphs:license"],
    default_visibility = ["//visibility:public"],
)

pytype_strict_library(
    name = "data",
    srcs = ["data.py"],
    srcs_version = "PY3ONLY",
    deps = [
        "//third_party/py/absl/flags",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_gnn",
        "//third_party/py/tqdm",
    ],
)

pytype_strict_library(
    name = "implicit",
    srcs = ["implicit.py"],
    srcs_version = "PY3ONLY",
    deps = [
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_gnn",
    ],
)

pytype_strict_test(
    name = "implicit_test",
    srcs = ["implicit_test.py"],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [
        ":implicit",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_gnn",
    ],
)

pytype_strict_library(
    name = "metrics",
    srcs = ["metrics.py"],
    srcs_version = "PY3ONLY",
    deps = [
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_ranking",
    ],
)

pytype_strict_library(
    name = "models",
    srcs = ["models.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":implicit",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_gnn",
    ],
)

pytype_strict_binary(
    name = "random_baseline",
    srcs = ["random_baseline.py"],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [
        ":data",
        ":metrics",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_binary(
    name = "train",
    srcs = ["train.py"],
    python_version = "PY3",
    srcs_version = "PY3ONLY",
    deps = [
        ":train_args",
        ":train_lib",
        "//third_party/py/absl:app",
    ],
)

pytype_strict_library(
    name = "train_args",
    srcs = ["train_args.py"],
    srcs_version = "PY3ONLY",
    deps = [
        "//third_party/py/absl/flags",
    ],
)

pytype_strict_library(
    name = "train_lib",
    srcs = ["train_lib.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":data",
        ":metrics",
        ":models",
        ":train_args",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_gnn",
        "//third_party/py/tqdm",
    ],
)
