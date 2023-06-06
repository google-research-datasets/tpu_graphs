#include "third_party/py/tpu_graphs/process_data/xla/featurizers.h"

#include <string>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/py/tpu_graphs/process_data/xla/hlo_opcode.h"
#include "third_party/tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "third_party/tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "third_party/tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace ml_lib {
namespace {

using ::testing::Each;
using ::testing::Eq;
using ::testing::Gt;
using ::testing::IsEmpty;
using ::testing::SizeIs;

class FeaturizersTest : public HloTestBase {};

TEST_F(FeaturizersTest, TestFeaturizeTileSizeConfigWithNullSourceIsZeroed) {
  std::vector<float> result;
  FeaturizeTileSizeConfig(&result, nullptr, FEATURE_WINDOW);
  EXPECT_THAT(result, SizeIs(Gt(0)));
  EXPECT_THAT(result, Each(Eq(0)));
}

TEST_F(FeaturizersTest, TestFeaturizeTileSizeConfigWhenWindowFeatureIsOff) {
  std::vector<float> result;
  FeaturizeTileSizeConfig(&result, nullptr, 0);
  EXPECT_THAT(result, IsEmpty());
}

TEST_F(FeaturizersTest, NodeFeatureLength) {
  // A convolution fusion op from mnasnet_b1_batch_128_df.
  const char* const hlo_string =
      R"(
HloModule DoIt_PadThenDynamicSlice, is_scheduled=true

%fused_computation (param_0.1: s32[], param_1.1: s32[], param_2.1: f32[7,7,128,128]) -> f32[4,4,128,128] {
  %param_2.1 = f32[7,7,128,128]{3,2,1,0} parameter(2)
  %constant.3 = f32[] constant(0)
  %pad.1 = f32[8,8,128,128]{3,2,1,0} pad(f32[7,7,128,128]{3,2,1,0} %param_2.1, f32[] %constant.3), padding=0_1x0_1x0_0x0_0
  %param_0.1 = s32[] parameter(0)
  %param_1.1 = s32[] parameter(1)
  ROOT %dynamic-slice.1 = f32[4,4,128,128]{3,2,1,0} dynamic-slice(f32[8,8,128,128]{3,2,1,0} %pad.1, s32[] %param_0.1, s32[] %param_0.1, s32[] %param_1.1, s32[] %param_1.1), dynamic_slice_sizes={4,4,128,128}
}

ENTRY %DoIt_PadThenDynamicSlice (A: f32[7,7,128,128]) -> f32[4,4,128,128] {
  %constant.2 = s32[] constant(0)
  %constant.1 = s32[] constant(4)
  %A = f32[7,7,128,128]{3,2,1,0} parameter(0)
  ROOT %fusion = f32[4,4,128,128]{3,2,1,0} fusion(s32[] %constant.1, s32[] %constant.2, f32[7,7,128,128]{3,2,1,0} %A), kind=kLoop, calls=%fused_computation
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_string));
  std::vector<std::string> node_feature_names;
  NamesForFeaturizeHloInstruction(&node_feature_names, 0, "");
  EXPECT_THAT(node_feature_names, SizeIs(kMinimalNodeFeatureCount));

  node_feature_names.clear();
  NamesForFeaturizeHloInstruction(
      &node_feature_names, FEATURE_OP_NON_ZERO | FEATURE_MODULE_NON_ZERO, "");
  EXPECT_THAT(node_feature_names, SizeIs(kMinimalNodeFeatureCount +
                                         kOpLevelNonZeroNodeFeatureCount +
                                         kModuleLevelNonZeroNodeFeatureCount));

  LOG(INFO) << "node features:";
  for (int i = 0; i < node_feature_names.size(); ++i) {
    LOG(INFO) << i << ": " << node_feature_names[i];
  }

  std::vector<std::string> tile_size_feature_names;
  NamesForFeaturizeTileSizeConfig(&tile_size_feature_names, FEATURE_WINDOW, "");
  EXPECT_THAT(tile_size_feature_names, SizeIs(kWindowConfigFeatureCount));

  LOG(INFO) << "Tile size config features:";
  for (int i = 0; i < tile_size_feature_names.size(); ++i) {
    LOG(INFO) << i << ": " << tile_size_feature_names[i];
  }

  const HloModuleProto proto = hlo_module->ToProto();
  for (const auto& computation : proto.computations()) {
    for (const auto& inst : computation.instructions()) {
      LOG(INFO) << inst.name();
      std::vector<float> node_feature;
      FeaturizeHloInstruction(&node_feature, inst, computation,
                              FEATURE_OP_NON_ZERO);
      EXPECT_THAT(node_feature, SizeIs(kMinimalNodeFeatureCount +
                                       kOpLevelNonZeroNodeFeatureCount));
    }
  }
}

}  // namespace
}  // namespace ml_lib
}  // namespace xla
