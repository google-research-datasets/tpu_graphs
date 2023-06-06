# tpu_graphs (go/tpu_graphs)
load("//tools/build_defs/license:license.bzl", "license")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")
load("//devtools/copybara/rules:copybara.bzl", "copybara_config_test")
load("//third_party/py/etils:build_defs.bzl", "glob_py_srcs")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:public"],
)

license(
    name = "license",
    package_name = "tpu_graphs",
    license_kinds = [
        "//devtools/compliance/licenses/spdx:Apache-2.0",
    ],
)

licenses(["notice"])

exports_files(["LICENSE"])

package_group(
    name = "internal",
    packages = [
        "//research/graph/datasets/tpu_graphs/...",
        "//third_party/py/tpu_graphs/...",
    ],
)

# tpu_graphs public API
# This is a single py_library rule which centralize all files/deps.
pytype_strict_library(
    name = "tpu_graphs",
    # Recursivelly auto-collect all `.py` files (excluding tests).
    # Note that `glob` won't recurse when subfolders have `BUILD` files. To have
    # a single top-level rule with additional `py_test` BUILD rules in subfolders, see:
    # go/oss-kit#single-rule-pattern.
    srcs = glob_py_srcs(),
    visibility = ["//visibility:public"],
    # Project dependencies (matching the `pip install` deps defined in `pyproject.toml`)
    deps = [
    ],
)

copybara_config_test(
    name = "copybara_test",
    config = "copy.bara.sky",
    deps = [
        "//third_party/py/etils:copybara_utils",
    ],
)
