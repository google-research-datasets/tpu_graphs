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
