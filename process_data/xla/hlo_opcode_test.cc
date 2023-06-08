/* Copyright 2023 Google LLC. All Rights Reserved.

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

#include "third_party/py/tpu_graphs/process_data/xla/hlo_opcode.h"

#include "testing/base/public/gunit.h"

namespace xla {
namespace ml_lib {
namespace {

TEST(HloOpcodeTest, ConsecutiveOpcodeIDs) {
  const uint64_t num_opcodes = NumOpcodes() + NumTombstoneOpcodes() + 1;
  EXPECT_GT(num_opcodes, 100);
  EXPECT_EQ(OpcodeIDToString(0), "unknown");
  EXPECT_EQ(OpcodeIDToString(num_opcodes), "unknown");

  for (uint64_t i = 1; i < num_opcodes; ++i) {
    const std::string opcode_str = OpcodeIDToString(i);
    EXPECT_NE(opcode_str, "unknown");
    EXPECT_EQ(StringToOpcodeID(opcode_str), i);
  }
}

TEST(HloOpcodeTest, UnknownOpHasIdZero) {
  EXPECT_EQ(StringToOpcodeID("abc"), 0);
  EXPECT_EQ(StringToOpcodeID("unknown"), 0);
}

}  // namespace
}  // namespace ml_lib
}  // namespace xla
