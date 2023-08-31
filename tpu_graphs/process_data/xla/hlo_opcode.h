/* Copyright 2023 The tpu_graphs Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_PY_TPU_GRAPHS_PROCESS_DATA_XLA_HLO_OPCODE_H_
#define THIRD_PARTY_PY_TPU_GRAPHS_PROCESS_DATA_XLA_HLO_OPCODE_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"

namespace xla {
namespace ml_lib {

#define FEATURE_WINDOW 1 << 0
#define FEATURE_OP_NON_ZERO 1 << 1
#define FEATURE_OP_ZERO 1 << 2
#define FEATURE_MODULE_NON_ZERO 1 << 3

// Node features.
const constexpr uint16_t kMinimalNodeFeatureCount = 113;
const constexpr uint16_t kOpLevelNonZeroNodeFeatureCount = 27;
const constexpr uint16_t kModuleLevelNonZeroNodeFeatureCount = 29;

// Module features.
const constexpr uint16_t kWindowConfigFeatureCount = 24;

// Config features.
const constexpr uint16_t kFusionConfigFeatureCount = 1;
const constexpr uint16_t kLayoutConfigFeatureCount = 18;
const constexpr uint16_t kDotConfigFeatureCount = 3;
// make sure to compute feature ranges after module features are finalized

inline uint8_t GetIncludeFeatureBits(absl::string_view task) {
  if (task == "op_window_cost") {
    return FEATURE_OP_NON_ZERO | FEATURE_WINDOW;
  }
  if (task == "module_fusion_cost" || task == "module_layout_cost" ||
      task == "module_dot_cost") {
    return FEATURE_OP_NON_ZERO;
  }
  return 0;
}

inline uint16_t GetNodeFeatureCount(absl::string_view task) {
  if (task == "op_window_cost") {
    return kMinimalNodeFeatureCount + kOpLevelNonZeroNodeFeatureCount;
  }
  if (task == "module_fusion_cost" || task == "module_layout_cost" ||
      task == "module_dot_cost") {
    return kMinimalNodeFeatureCount + kOpLevelNonZeroNodeFeatureCount;
  }
  return kMinimalNodeFeatureCount;
}

inline uint16_t GetModuleFeatureCount(absl::string_view task) {
  if (task == "op_window_cost") {
    return kWindowConfigFeatureCount;
  }
  return 0;
}

inline uint16_t GetConfigFeatureCount(absl::string_view task) {
  if (task == "module_fusion_cost") {
    return kFusionConfigFeatureCount;
  }
  if (task == "module_layout_cost") {
    return kLayoutConfigFeatureCount;
  }
  if (task == "module_dot_cost") {
    return kDotConfigFeatureCount;
  }
  return 0;
}

// List of opcodes and IDs. Maintain our own list instead of relying XLA to
// guarantee stable opcode IDs.  Reserve ID 0 for unknown opcodes.
//
// CAUTION: New opcodes must be appended to the end of this list with increasing
// consecutive IDs. The existing opcodes and IDs must never be modified. Doing
// so will mess up any machine learned model.
#define LEARNED_HLO_OPCODE_LIST(V)   \
  V("abs", 1)                        \
  V("add", 2)                        \
  V("add-dependency", 3)             \
  V("after-all", 4)                  \
  V("all-reduce", 5)                 \
  V("all-to-all", 6)                 \
  V("atan2", 7)                      \
  V("batch-norm-grad", 8)            \
  V("batch-norm-inference", 9)       \
  V("batch-norm-training", 10)       \
  V("bitcast", 11)                   \
  V("bitcast-convert", 12)           \
  V("broadcast", 13)                 \
  V("call", 14)                      \
  V("ceil", 15)                      \
  V("cholesky", 16)                  \
  V("clamp", 17)                     \
  V("collective-permute", 18)        \
  V("count-leading-zeros", 19)       \
  V("compare", 20)                   \
  V("complex", 21)                   \
  V("concatenate", 22)               \
  V("conditional", 23)               \
  V("constant", 24)                  \
  V("convert", 25)                   \
  V("convolution", 26)               \
  V("copy", 27)                      \
  V("copy-done", 28)                 \
  V("copy-start", 29)                \
  V("cosine", 30)                    \
  V("custom-call", 31)               \
  V("divide", 32)                    \
  V("domain", 33)                    \
  V("dot", 34)                       \
  V("dynamic-slice", 35)             \
  V("dynamic-update-slice", 36)      \
  V("exponential", 37)               \
  V("exponential-minus-one", 38)     \
  V("fft", 39)                       \
  V("floor", 40)                     \
  V("fusion", 41)                    \
  V("gather", 42)                    \
  V("get-dimension-size", 43)        \
  V("set-dimension-size", 44)        \
  V("get-tuple-element", 45)         \
  V("imag", 46)                      \
  V("infeed", 47)                    \
  V("iota", 48)                      \
  V("is-finite", 49)                 \
  V("log", 50)                       \
  V("log-plus-one", 51)              \
  V("and", 52)                       \
  V("not", 53)                       \
  V("or", 54)                        \
  V("xor", 55)                       \
  V("map", 56)                       \
  V("maximum", 57)                   \
  V("minimum", 58)                   \
  V("multiply", 59)                  \
  V("negate", 60)                    \
  V("outfeed", 61)                   \
  V("pad", 62)                       \
  V("parameter", 63)                 \
  V("partition-id", 64)              \
  V("popcnt", 65)                    \
  V("power", 66)                     \
  V("real", 67)                      \
  V("recv", 68)                      \
  V("recv-done", 69)                 \
  V("reduce", 70)                    \
  V("reduce-precision", 71)          \
  V("reduce-window", 72)             \
  V("remainder", 73)                 \
  V("replica-id", 74)                \
  V("reshape", 75)                   \
  V("reverse", 76)                   \
  V("rng", 77)                       \
  V("rng-get-and-update-state", 78)  \
  V("rng-bit-generator", 79)         \
  V("round-nearest-afz", 80)         \
  V("rsqrt", 81)                     \
  V("scatter", 82)                   \
  V("select", 83)                    \
  V("select-and-scatter", 84)        \
  V("send", 85)                      \
  V("send-done", 86)                 \
  V("shift-left", 87)                \
  V("shift-right-arithmetic", 88)    \
  V("shift-right-logical", 89)       \
  V("sign", 90)                      \
  V("sine", 91)                      \
  V("slice", 92)                     \
  V("sort", 93)                      \
  V("sqrt", 94)                      \
  V("subtract", 95)                  \
  V("tanh", 96)                      \
  V("transpose", 98)                 \
  V("triangular-solve", 99)          \
  V("tuple", 100)                    \
  V("while", 102)                    \
  V("cbrt", 103)                     \
  V("all-gather", 104)               \
  V("collective-permute-start", 105) \
  V("collective-permute-done", 106)  \
  V("logistic", 107)                 \
  V("dynamic-reshape", 108)          \
  V("all-reduce-start", 109)         \
  V("all-reduce-done", 110)          \
  V("reduce-scatter", 111)           \
  V("all-gather-start", 112)         \
  V("all-gather-done", 113)          \
  V("opt-barrier", 114)              \
  V("async-start", 115)              \
  V("async-update", 116)             \
  V("async-done", 117)               \
  V("round-nearest-even", 118)       \
  V("stochastic-convert", 119)       \
  V("tan", 120)

// Ops that have been removed from HLO
#define TOMBSTONE_HLO_OPCODE_LIST(V) \
  V("trace", 97)                     \
  V("tuple-select", 101)

// CAUTION: New opcodes must be appended to the end of this list with increasing
// consecutive IDs. The existing opcodes and IDs must never be modified. Doing
// so will mess up any machine learned model.

// Converts opcode string into opcode ID. Returns 0 for unknown opcode.
inline uint32_t StringToOpcodeID(absl::string_view opcode_name) {
  static auto* opcode_map = new absl::flat_hash_map<std::string, uint32_t>({
#define STRING_TO_OPCODE_ENTRY(opcode_name, opcode_id) {opcode_name, opcode_id},
      LEARNED_HLO_OPCODE_LIST(STRING_TO_OPCODE_ENTRY)
          TOMBSTONE_HLO_OPCODE_LIST(STRING_TO_OPCODE_ENTRY)
#undef STRING_TO_OPCODE_ENTRY
  });
  auto it = opcode_map->find(opcode_name);
  if (it == opcode_map->end()) {
    return 0;
  }
  return it->second;
}

// Converts opcode ID into opcode string. Returns unknown for unknown IDs.
inline std::string OpcodeIDToString(uint64_t opcode_id) {
  static auto* opcode_map = new absl::flat_hash_map<uint32_t, std::string>({
#define STRING_TO_OPCODE_ENTRY(opcode_name, opcode_id) {opcode_id, opcode_name},
      LEARNED_HLO_OPCODE_LIST(STRING_TO_OPCODE_ENTRY)
          TOMBSTONE_HLO_OPCODE_LIST(STRING_TO_OPCODE_ENTRY)
#undef STRING_TO_OPCODE_ENTRY
  });
  auto it = opcode_map->find(opcode_id);
  if (it == opcode_map->end()) {
    return "unknown";
  }
  return it->second;
}

inline const uint32_t NumOpcodes() {
#define HLO_COUNT_ONE(...) +1
#define HLO_XLIST_LENGTH(list) list(HLO_COUNT_ONE)
  return HLO_XLIST_LENGTH(LEARNED_HLO_OPCODE_LIST);
}

inline const uint32_t NumTombstoneOpcodes() {
  return HLO_XLIST_LENGTH(TOMBSTONE_HLO_OPCODE_LIST);
}

inline const bool IsTombstone(uint64_t opcode_id) {
#define HLO_IN_ONE(name, id) opcode_id == id ||
#define HLO_XLIST_CONTAINS(list) list(HLO_IN_ONE) false;
  return HLO_XLIST_CONTAINS(TOMBSTONE_HLO_OPCODE_LIST);
#undef HLO_XLIST_CONTAINS
#undef HLO_IN_ONE
}

}  // namespace ml_lib
}  // namespace xla

#endif  // THIRD_PARTY_PY_TPU_GRAPHS_PROCESS_DATA_XLA_HLO_OPCODE_H_
