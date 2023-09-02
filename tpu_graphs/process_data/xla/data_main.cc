#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "tpu_graphs/process_data/xla/featurizers.h"
#include "tpu_graphs/process_data/xla/hlo_encoder.h"
#include "tpu_graphs/process_data/xla/hlo_opcode.h"
#include "tpu_graphs/process_data/xla/tuning_data_iterator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/env.h"

namespace xla {
namespace ml_lib {

namespace tf = ::tensorflow;

void TestFeaturizeTileSizeConfig() {
  std::cout << "TestFeaturizeTileSizeConfig\n";
  std::vector<float> result;
  FeaturizeTileSizeConfig(&result, nullptr, FEATURE_WINDOW);
  std::cout << "Size: " << result.size() << "\n";
}

void TestDenseNeighborIndicesBuilderOnOneGraphOneNode() {
  std::cout << "TestDenseNeighborIndicesBuilderOnOneGraphOneNode\n";
  std::vector<std::vector<int64_t>> identifiers = {{0}};
  tf::Tensor module_ids(tf::DT_INT64, tf::TensorShape({1}));
  tf::Tensor seq_indices(tf::DT_INT32, tf::TensorShape({1, 1}));
  tf::Tensor seq_masks(tf::DT_BOOL, tf::TensorShape({1, 1}));
  tf::Tensor neighbor_indices(tf::DT_INT64, tf::TensorShape({1, 1}));
  tf::Tensor neighbor_masks(tf::DT_FLOAT, tf::TensorShape({1, 1}));
  DenseNeighborIndicesBuilder builder(identifiers,
                                      /*module_count=*/1, &module_ids,
                                      &seq_indices, &seq_masks,
                                      &neighbor_indices, &neighbor_masks);
  builder.AdvanceModule();
  builder.AdvanceNode();
  CHECK(builder.Finalize().ok());
  CHECK_EQ(module_ids.flat<int64_t>()(0), 0);
  CHECK_EQ(seq_indices.matrix<int32_t>()(0, 0), 0);
  CHECK_EQ(seq_masks.matrix<bool>()(0, 0), 1);
  CHECK_EQ(neighbor_indices.matrix<int64_t>()(0, 0), 0);
  CHECK_EQ(neighbor_masks.matrix<float>()(0, 0), 0);
}

void TestModuleTuningDataNoFilter(const std::string& source_path) {
  std::cout << "TestModuleTuningDataNoFilter\n";
  tpu_graphs::ModuleTuningData proto_data;
  CHECK(tf::ReadBinaryProto(tf::Env::Default(), source_path, &proto_data).ok());

  std::unique_ptr<TuningDataIterator> data_iterator =
      CreateTuningDataIterator(kModuleTuning, source_path).value();
  CHECK(data_iterator != nullptr);
  std::cout << "Size: " << data_iterator->Name() << "\n";
  std::cout << "Sample count: " << data_iterator->GetSampleCount() << "\n";
}

}  // namespace ml_lib
}  // namespace xla

int main(int argc, char* argv[]) {
  xla::ml_lib::TestFeaturizeTileSizeConfig();
  xla::ml_lib::TestDenseNeighborIndicesBuilderOnOneGraphOneNode();
  xla::ml_lib::TestModuleTuningDataNoFilter(argv[1]);
  return 0;
}
