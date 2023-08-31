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

#ifndef THIRD_PARTY_PY_TPU_GRAPHS_PROCESS_DATA_XLA_FEATURIZERS_H_
#define THIRD_PARTY_PY_TPU_GRAPHS_PROCESS_DATA_XLA_FEATURIZERS_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "google/protobuf/repeated_field.h"
#include "tpu_graphs/process_data/xla/hlo_opcode.h"
#include "tpu_graphs/proto/tuning.pb.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace ml_lib {

static const std::vector<
    std::pair<const PrimitiveType, const absl::string_view>>
    kPrimitiveTypeFeatures = {{PRIMITIVE_TYPE_INVALID, "is_invalid_type"},
                              {PRED, "is_pred"},
                              {S8, "is_s8"},
                              {S16, "is_s16"},
                              {S32, "is_s32"},
                              {S64, "is_s64"},
                              {U8, "is_u8"},
                              {U16, "is_u16"},
                              {U32, "is_u32"},
                              {U64, "is_u64"},
                              {F16, "is_f16"},
                              {F32, "is_f32"},
                              {F64, "is_f64"},
                              {BF16, "is_bf16"},
                              {C64, "is_c64"},
                              {C128, "is_c128"},
                              {TUPLE, "is_tuple"},
                              {OPAQUE_TYPE, "is_opaque_type"},
                              {TOKEN, "is_token"}};

template <uint8_t LeadingNumbers, typename Dest>
void NamesForFixedNumericSequenceFeatures(Dest* out,
                                          const std::string& prefix) {
  for (int idx = 0; idx < LeadingNumbers; ++idx) {
    out->push_back(absl::StrFormat("%s_%d", prefix, idx));
  }
  out->push_back(absl::StrCat(prefix, "_sum"));
  out->push_back(absl::StrCat(prefix, "_product"));
}

// Pushes `LeadingNumbers + 2` floating-point features suitable for input into a
// machine learning model. These features are derived from an any-length
// iterable `source` which is converted elementwise into integers via `mapper`.
template <uint8_t LeadingNumbers, typename Container, typename Dest>
void FixedNumericSequenceFeatures(
    Dest* out, const Container& source,
    std::function<int64_t(decltype(source.at(0)))> mapper,
    bool zero_product_on_empty_source = false) {
  int64_t forwarded = 0;
  int64_t sum = 0;
  int64_t product = 1;
  bool source_was_empty = true;
  for (auto& val : source) {
    source_was_empty = false;
    const int64_t mapped_val = mapper(val);
    sum += mapped_val;
    product *= mapped_val;
    if (forwarded < LeadingNumbers) {
      out->push_back(static_cast<float>(mapped_val));
      forwarded++;
    }
  }
  for (; forwarded < LeadingNumbers; ++forwarded) {
    out->push_back(0.);
  }
  out->push_back(static_cast<float>(sum));
  if (source_was_empty && zero_product_on_empty_source) {
    out->push_back(0.0f);
  } else {
    out->push_back(static_cast<float>(product));
  }
}

// Same as above but consumes a `source` of integers and is equivalent using the
// the identity function as the mapper.
template <uint8_t LeadingNumbers, typename Container, typename Dest>
void FixedNumericSequenceFeatures(Dest* out, const Container& source,
                                  bool zero_product_on_empty_source = false) {
  FixedNumericSequenceFeatures<LeadingNumbers, Container, Dest>(
      out, source, [](int64_t x) -> int64_t { return x; },
      /*zero_product_on_empty_source=*/zero_product_on_empty_source);
}

template <uint8_t LeadingNumbers, typename Dest>
void NamesForFixedNumericSequenceFeaturesNoAggregate(
    Dest* out, const std::string& prefix) {
  for (int idx = 0; idx < LeadingNumbers; ++idx) {
    out->push_back(absl::StrFormat("%s_%d", prefix, idx));
  }
}

// Similar to `FixedNumericSequenceFeatures` but don't produce sum and product.
template <uint8_t LeadingNumbers, typename Container, typename Dest>
void FixedNumericSequenceFeaturesNoAggregate(
    Dest* out, const Container& source,
    std::function<int64_t(decltype(source.at(0)))> mapper) {
  int64_t forwarded = 0;
  bool source_was_empty = true;
  for (auto& val : source) {
    source_was_empty = false;
    const int64_t mapped_val = mapper(val);
    if (forwarded < LeadingNumbers) {
      out->push_back(static_cast<float>(mapped_val));
      forwarded++;
    }
  }
  for (; forwarded < LeadingNumbers; ++forwarded) {
    out->push_back(0.);
  }
}

// Same as above but consumes a `source` of integers and is equivalent using the
// the identity function as the mapper.
template <uint8_t LeadingNumbers, typename Container, typename Dest>
void FixedNumericSequenceFeaturesNoAggregate(Dest* out,
                                             const Container& source) {
  FixedNumericSequenceFeaturesNoAggregate<LeadingNumbers, Container, Dest>(
      out, source, [](int64_t x) -> int64_t { return x; });
}

template <uint8_t LeadingNumbers, typename Dest>
void NamesForFeaturizeVariableLengthBools(Dest* out,
                                          const std::string& prefix) {
  for (int idx = 0; idx < LeadingNumbers; ++idx) {
    out->push_back(absl::StrFormat("%s_%d", prefix, idx));
  }
  out->push_back(absl::StrCat(prefix, "_true_count"));
  out->push_back(absl::StrCat(prefix, "_false_count"));
}

// Push features into `out` corresponding to the booleans in `source` suitable
// for input into a machine learning model.
template <uint8_t LeadingNumbers, typename Dest, typename Container>
void FeaturizeVariableLengthBools(
    Dest* out, const Container& source,
    std::function<bool(decltype(source.at(0)))> mapper) {
  int64_t forwarded = 0;
  int64_t positives = 0;
  int64_t negatives = 0;
  for (const auto& val : source) {
    bool b = mapper(val);
    if (forwarded < LeadingNumbers) {
      out->push_back(b ? 1. : 0.);
      forwarded++;
    }
    if (b) {
      positives += 1;
    } else {
      negatives += 1;
    }
  }
  for (; forwarded < LeadingNumbers; ++forwarded) {
    out->push_back(0.);
  }
  out->push_back(static_cast<float>(positives));
  out->push_back(static_cast<float>(negatives));
}

// Same as above but consumes a `source` of bool and is equivalent using the
// the identity function as the mapper.
template <uint8_t LeadingNumbers, typename Dest, typename Container>
void FeaturizeVariableLengthBools(Dest* out, const Container& source) {
  FeaturizeVariableLengthBools<LeadingNumbers, Dest, Container>(
      out, source, [](bool x) -> bool { return x; });
}

// This macro produces two function definitions:
//  (a) a function `fn_name` which pushes features into the container passed as
//      its first argument `out` and accepts one arbitrary argument
//      `source_arg`,
//  (b) a function named NamesFor`fn_name` (the same name with a "NamesFor"
//      prefix) which pushes strings corresponding to feature names into its
//      first argument `out` and accepts and optional parameter `prefix`.
#define FEATURIZE_FUNCTION_PAIR(fn_name, source_arg, ...)                 \
  template <bool yield_names, typename Dest>                              \
  void _Inner_##fn_name(Dest* out, const source_arg* source_ptr,          \
                        const uint8_t include_features,                   \
                        const std::string& prefix) {                      \
    const source_arg& source = *source_ptr;                               \
    __VA_ARGS__                                                           \
  }                                                                       \
  template <typename Dest>                                                \
  void(fn_name)(Dest * out, const source_arg& source,                     \
                const uint8_t include_features) {                         \
    _Inner_##fn_name<false, Dest>(out, &source, include_features, "");    \
  }                                                                       \
  template <typename Dest>                                                \
  void(NamesFor##fn_name)(Dest * out, const uint8_t include_features,     \
                          const std::string& prefix = "") {               \
    source_arg source;                                                    \
    _Inner_##fn_name<true, Dest>(out, &source, include_features, prefix); \
  }

// Identical to `FEATURIZE_FUNCTION_PAIR` but the function `fn_name` accepts two
// arguments `source_arg1` and `source_arg2`.
#define FEATURIZE_FUNCTION_PAIR_2ARG(fn_name, source_arg1, source_arg2, ...)  \
  template <bool yield_names, typename Dest>                                  \
  void _Inner_##fn_name(Dest* out, const source_arg1* source_ptr1,            \
                        const source_arg2* source_ptr2,                       \
                        const uint8_t include_features,                       \
                        const std::string& prefix) {                          \
    const source_arg1& source1 = *source_ptr1;                                \
    const source_arg2& source2 = *source_ptr2;                                \
    __VA_ARGS__                                                               \
  }                                                                           \
  template <typename Dest>                                                    \
  void(fn_name)(Dest * out, const source_arg1& source1,                       \
                const source_arg2& source2, const uint8_t include_features) { \
    _Inner_##fn_name<false, Dest>(out, &source1, &source2, include_features,  \
                                  "");                                        \
  }                                                                           \
  template <typename Dest>                                                    \
  void(NamesFor##fn_name)(Dest * out, const uint8_t include_features,         \
                          const std::string& prefix = "") {                   \
    source_arg1 source1;                                                      \
    source_arg2 source2;                                                      \
    _Inner_##fn_name<true, Dest>(out, &source1, &source2, include_features,   \
                                 prefix);                                     \
  }

// Same as above, but for functions which consumes pointers to their operand.
#define FEATURIZE_FUNCTION_PAIR_PTR(fn_name, source_arg, ...)             \
  template <bool yield_names, typename Dest>                              \
  void _Inner_##fn_name(Dest* out, const source_arg* source,              \
                        const uint8_t include_features,                   \
                        const std::string& prefix) {                      \
    __VA_ARGS__                                                           \
  }                                                                       \
  template <typename Dest>                                                \
  void(fn_name)(Dest * out, const source_arg* source,                     \
                const uint8_t include_features) {                         \
    _Inner_##fn_name<false, Dest>(out, source, include_features, "");     \
  }                                                                       \
  template <typename Dest>                                                \
  void(NamesFor##fn_name)(Dest * out, const uint8_t include_features,     \
                          const std::string& prefix = "") {               \
    source_arg source;                                                    \
    _Inner_##fn_name<true, Dest>(out, &source, include_features, prefix); \
  }

// This returns either `expr` or the string `name` with a prefix prepended.
//
// This macro is meant to be used within the body of a function defined with
// FEATURIZE_FUNCTION_PAIR or its variants. It may consume arguments defined by
// that macro (e.g. the prefix).
#define FEATURIZE_SWITCH(name, expr)                           \
  ([&]() {                                                     \
    if constexpr (yield_names) {                               \
      const auto adjusted_prefix =                             \
          (prefix).empty() ? "" : absl::StrCat((prefix), "_"); \
      return absl::StrCat(adjusted_prefix, (name));            \
    } else {                                                   \
      return (expr);                                           \
    }                                                          \
  })()

// This will call either `fn_name` or its NamesFor... variant with extended
// prefix `p`, corresponding to the calling context.
#define FEATURIZE_DISPATCH(fn_name, p, ...)                       \
  if constexpr (yield_names) {                                    \
    const auto new_prefix =                                       \
        prefix.empty() ? p : absl::StrFormat("%s_%s", prefix, p); \
    NamesFor##fn_name(out, new_prefix);                           \
  } else {                                                        \
    fn_name(out, __VA_ARGS__);                                    \
  }

// This will call either `fn_name` or its NamesFor... variant with extended
// prefix `p`, corresponding to the calling context.
#define FEATURIZE_DISPATCH_COND(fn_name, include_features, p, ...) \
  if constexpr (yield_names) {                                     \
    const auto new_prefix =                                        \
        prefix.empty() ? p : absl::StrFormat("%s_%s", prefix, p);  \
    NamesFor##fn_name(out, include_features, new_prefix);          \
  } else {                                                         \
    fn_name(out, __VA_ARGS__, include_features);                   \
  }

// Turns an `xla::ConvolutionDimensionNumbers` protobuf into a vector of
// features suitable for a machine learning model.
//
// This macro is meant to be used within the body of a function defined with
// FEATURIZE_FUNCTION_PAIR or its variants. It may consume arguments defined by
// that macro (e.g. the prefix).
FEATURIZE_FUNCTION_PAIR(
    FeaturizeConvolutionDimensionNumbers, ConvolutionDimensionNumbers, {
      out->push_back(
          FEATURIZE_SWITCH("input_batch_dim", source.input_batch_dimension()));
      out->push_back(FEATURIZE_SWITCH("input_feature_dim",
                                      source.input_feature_dimension()));
      FEATURIZE_DISPATCH(FixedNumericSequenceFeaturesNoAggregate<4>,
                         "input_spatial_dims",
                         source.input_spatial_dimensions());
      out->push_back(FEATURIZE_SWITCH("kernel_input_feature_dim",
                                      source.kernel_input_feature_dimension()));
      out->push_back(
          FEATURIZE_SWITCH("kernel_output_feature_dim",
                           source.kernel_output_feature_dimension()));
      FEATURIZE_DISPATCH(FixedNumericSequenceFeaturesNoAggregate<4>,
                         "kernel_spatial_dims",
                         source.kernel_spatial_dimensions());
      out->push_back(FEATURIZE_SWITCH("output_batch_dim",
                                      source.output_batch_dimension()));
      out->push_back(FEATURIZE_SWITCH("output_feature_dim",
                                      source.output_feature_dimension()));
      // No need to include output spatial dimensions because they can be
      // inferred.
    })

// Turns a `google::protobuf::RepeatedPtrField` of `xla::SliceDimensions` protobufs into a
// vector of features suitable for input to a machine learning model.
FEATURIZE_FUNCTION_PAIR(
    FeaturizeSliceDimensions,
    google::protobuf::RepeatedPtrField<HloInstructionProto::SliceDimensions>, {
      FEATURIZE_DISPATCH(
          FixedNumericSequenceFeatures<2>, "start", source,
          [](const HloInstructionProto::SliceDimensions& s) {
            return s.start();
          },
          /*zero_product_on_empty_source=*/false);
      FEATURIZE_DISPATCH(
          FixedNumericSequenceFeatures<2>, "stride", source,
          [](const HloInstructionProto::SliceDimensions& s) {
            return s.stride();
          },
          /*zero_product_on_empty_source=*/false);
      FEATURIZE_DISPATCH(
          FixedNumericSequenceFeatures<2>, "limit", source,
          [](const HloInstructionProto::SliceDimensions& s) {
            return s.limit();
          },
          /*zero_product_on_empty_source=*/false);
    })

// Turns an `xla::PaddingConfig` protobuf into an inlined vector of features
// suitable for input to a machine learning model.
FEATURIZE_FUNCTION_PAIR(FeaturizePaddingConfig, PaddingConfig, {
  FEATURIZE_DISPATCH(
      FixedNumericSequenceFeatures<2>, "edge_padding_low", source.dimensions(),
      [](const PaddingConfig::PaddingConfigDimension& s) {
        return s.edge_padding_low();
      },
      /*zero_product_on_empty_source=*/false);
  FEATURIZE_DISPATCH(
      FixedNumericSequenceFeatures<2>, "edge_padding_high", source.dimensions(),
      [](const PaddingConfig::PaddingConfigDimension& s) {
        return s.edge_padding_high();
      },
      /*zero_product_on_empty_source=*/false);
  if (include_features & FEATURE_OP_ZERO) {
    FEATURIZE_DISPATCH(
        FixedNumericSequenceFeatures<2>, "dims_interior_padding",
        source.dimensions(),
        [](const PaddingConfig::PaddingConfigDimension& s) {
          return s.interior_padding();
        },
        /*zero_product_on_empty_source=*/false);
  }
})

// Turns an `xla::DotDimensionNumbers` protobuf into an inlined vector of
// features suitable for input to a machine learning model.
FEATURIZE_FUNCTION_PAIR(FeaturizeDotDimensionNumbers, DotDimensionNumbers, {
  FEATURIZE_DISPATCH(FixedNumericSequenceFeaturesNoAggregate<2>,
                     "lhs_contracting_dims",
                     source.lhs_contracting_dimensions());
  FEATURIZE_DISPATCH(FixedNumericSequenceFeaturesNoAggregate<2>,
                     "rhs_contracting_dims",
                     source.rhs_contracting_dimensions());
  FEATURIZE_DISPATCH(FixedNumericSequenceFeaturesNoAggregate<2>,
                     "lhs_batch_dims", source.lhs_batch_dimensions());
  FEATURIZE_DISPATCH(FixedNumericSequenceFeaturesNoAggregate<2>,
                     "rhs_batch_dims", source.rhs_batch_dimensions());
})

// Pushes features suitable for input to a machine learning model into the
// vector `out` using the given `xla::LayoutProto`.
FEATURIZE_FUNCTION_PAIR(FeaturizeLayoutProto, LayoutProto, {
  FEATURIZE_DISPATCH(FixedNumericSequenceFeaturesNoAggregate<6>,
                     "minor_to_major", source.minor_to_major());
  if (include_features & FEATURE_OP_ZERO) {
    // TODO: may include `.tiles`
    out->push_back(FEATURIZE_SWITCH("element_size_in_bits", 0));
    out->push_back(FEATURIZE_SWITCH("memory_space", source.memory_space()));
  }
})

// Turns an `xla::ScatterDimensionNumbers` protobuf into a vector of features
// suitable for input to a machine learning model.
FEATURIZE_FUNCTION_PAIR(
    FeaturizeScatterDimensionNumbers, ScatterDimensionNumbers, {
      FEATURIZE_DISPATCH(FixedNumericSequenceFeaturesNoAggregate<3>,
                         "update_window_dims", source.update_window_dims());
      FEATURIZE_DISPATCH(FixedNumericSequenceFeaturesNoAggregate<3>,
                         "inserted_window_dims", source.inserted_window_dims());
      FEATURIZE_DISPATCH(FixedNumericSequenceFeaturesNoAggregate<3>,
                         "scatter_dims_to_operand_dims",
                         source.scatter_dims_to_operand_dims());
      out->push_back(
          FEATURIZE_SWITCH("index_vector_dim", source.index_vector_dim()));
    })

// Turns an `xla::GatherDimensionNumbers` protobuf into a vector of features
// suitable for input to a machine learning model.
FEATURIZE_FUNCTION_PAIR(
    FeaturizeGatherDimensionNumbers, GatherDimensionNumbers, {
      FEATURIZE_DISPATCH(FixedNumericSequenceFeaturesNoAggregate<3>,
                         "offset_dims", source.offset_dims());
      FEATURIZE_DISPATCH(FixedNumericSequenceFeaturesNoAggregate<3>,
                         "collapsed_slice_dims ",
                         source.collapsed_slice_dims());
      FEATURIZE_DISPATCH(FixedNumericSequenceFeaturesNoAggregate<3>,
                         "start_index_map", source.start_index_map());
      out->push_back(
          FEATURIZE_SWITCH("index_vector_dim", source.index_vector_dim()));
    })

// Converts an `xla::FftType` into a one-hot inlined vector.
FEATURIZE_FUNCTION_PAIR(FeaturizeFftType, FftType, {
  out->push_back(FEATURIZE_SWITCH("is_fft", source == FFT ? 1. : 0.));
  out->push_back(FEATURIZE_SWITCH("is_ifft", source == IFFT ? 1. : 0.));
  out->push_back(FEATURIZE_SWITCH("is_rfft", source == RFFT ? 1. : 0.));
  out->push_back(FEATURIZE_SWITCH("is_irfft", source == IRFFT ? 1. : 0.));
})

// Turns an `xla::Window` protobuf into a vector of features suitable for input
// to a machine learning model.
FEATURIZE_FUNCTION_PAIR(FeaturizeWindow, Window, {
  const auto& dims = source.dimensions();
  FEATURIZE_DISPATCH(
      FixedNumericSequenceFeatures<6>, "size", dims,
      [](const WindowDimension& d) { return d.size(); },
      /*zero_product_on_empty_source=*/false);
  FEATURIZE_DISPATCH(
      FixedNumericSequenceFeatures<6>, "stride", dims,
      [](const WindowDimension& d) { return d.stride(); },
      /*zero_product_on_empty_source=*/false);
  FEATURIZE_DISPATCH(
      FixedNumericSequenceFeatures<6>, "padding_low", dims,
      [](const WindowDimension& d) { return d.padding_low(); },
      /*zero_product_on_empty_source=*/false);
  FEATURIZE_DISPATCH(
      FixedNumericSequenceFeatures<6>, "padding_high", dims,
      [](const WindowDimension& d) { return d.padding_high(); },
      /*zero_product_on_empty_source=*/false);
  FEATURIZE_DISPATCH(
      FixedNumericSequenceFeatures<6>, "window_dilation", dims,
      [](const WindowDimension& d) { return d.window_dilation(); },
      /*zero_product_on_empty_source=*/false);
  FEATURIZE_DISPATCH(
      FixedNumericSequenceFeatures<6>, "base_dilation", dims,
      [](const WindowDimension& d) { return d.base_dilation(); },
      /*zero_product_on_empty_source=*/false);
  FEATURIZE_DISPATCH(
      FeaturizeVariableLengthBools<6>, "window_reversal", dims,
      [](const WindowDimension& d) { return d.window_reversal(); });
})

// Turns an `tpu_graphs::TileSizeConfig` protobuf into a vector of features
// suitable for input to a machine learning model.
FEATURIZE_FUNCTION_PAIR_PTR(
    FeaturizeTileSizeConfig, tpu_graphs::TileSizeConfig, {
      if (include_features & FEATURE_WINDOW) {
        std::unique_ptr<tpu_graphs::TileSizeConfig> zeroed_config = nullptr;
        if (!yield_names && source == nullptr) {
          zeroed_config = std::make_unique<tpu_graphs::TileSizeConfig>();
          source = zeroed_config.get();
        }

        FEATURIZE_DISPATCH(FixedNumericSequenceFeatures<6>, "kernel_bounds",
                           source->kernel_bounds(),
                           /*zero_product_on_empty_source=*/true);
        FEATURIZE_DISPATCH(FixedNumericSequenceFeatures<6>, "output_bounds",
                           source->output_bounds(),
                           /*zero_product_on_empty_source=*/true);
        FEATURIZE_DISPATCH(FixedNumericSequenceFeatures<6>, "input_bounds",
                           source->input_bounds(),
                           /*zero_product_on_empty_source=*/true);
      }
    })

// Pushes features suitable for input to a machine learning model into `out`
// describing the given `xla::PrimitiveType`. In particular, a one-hot encoding
// is used.
FEATURIZE_FUNCTION_PAIR(FeaturizePrimitiveType, PrimitiveType, {
  for (int idx = 0; idx < kPrimitiveTypeFeatures.size(); ++idx) {
    const auto& pair = kPrimitiveTypeFeatures[idx];
    out->push_back(
        FEATURIZE_SWITCH(pair.second, (pair.first == source ? 1. : 0)));
  }
})

// Turns an `xla::ShapeProto` protobuf into a vector of features suitable for a
// machine learning model.
FEATURIZE_FUNCTION_PAIR(FeaturizeShape, ShapeProto, {
  FEATURIZE_DISPATCH_COND(FeaturizePrimitiveType, include_features,
                          "element_type", source.element_type());
  FEATURIZE_DISPATCH(FixedNumericSequenceFeatures<6>, "dimensions",
                     source.dimensions());
  // Encode only the top-level length of `tuple_shapes`.
  out->push_back(FEATURIZE_SWITCH(
      "tuple_shapes_size", static_cast<float>(source.tuple_shapes_size())));
  if (include_features & FEATURE_OP_ZERO) {
    FEATURIZE_DISPATCH(FeaturizeVariableLengthBools<6>, "is_dynamic_dim",
                       source.is_dynamic_dimension());
  }
})

FEATURIZE_FUNCTION_PAIR_2ARG(
    FeaturizeHloInstruction, HloInstructionProto, HloComputationProto, {
      // Add a feature to indicate whether or not this instruction is the
      // computation's root instruction
      out->push_back(FEATURIZE_SWITCH(
          "is_root", source1.id() == source2.root_id() ? 1. : 0.));
      out->push_back(FEATURIZE_SWITCH("element_size_in_bits", 0));

      // Note that we don't encode `.literal` or compute time estimates
      FEATURIZE_DISPATCH_COND(FeaturizeShape, include_features, "shape",
                              source1.shape());
      out->push_back(FEATURIZE_SWITCH(
          "parameter_number", static_cast<float>(source1.parameter_number())));

      // Dimensions to be transposed, broadcasted, etc.
      FEATURIZE_DISPATCH(FixedNumericSequenceFeaturesNoAggregate<6>,
                         "dimensions", source1.dimensions());

      // Inapplicable fields, such as convolution dimension numbers for
      // non-convolution ops, are zero.
      FEATURIZE_DISPATCH_COND(FeaturizeWindow, include_features, "window",
                              source1.window());
      FEATURIZE_DISPATCH_COND(FeaturizeConvolutionDimensionNumbers,
                              include_features, "convolution_dim_numbers",
                              source1.convolution_dimension_numbers());

      if (include_features & FEATURE_OP_NON_ZERO) {
        out->push_back(FEATURIZE_SWITCH(
            "feature_group_count",
            static_cast<float>(source1.feature_group_count())));
        out->push_back(
            FEATURIZE_SWITCH("batch_group_count",
                             static_cast<float>(source1.batch_group_count())));
        FEATURIZE_DISPATCH_COND(FeaturizeSliceDimensions, include_features,
                                "slice_dims", source1.slice_dimensions());
        FEATURIZE_DISPATCH(FixedNumericSequenceFeatures<2>,
                           "dynamic_slice_sizes",
                           source1.dynamic_slice_sizes());
        FEATURIZE_DISPATCH_COND(FeaturizePaddingConfig, include_features,
                                "padding_config", source1.padding_config());
        out->push_back(
            FEATURIZE_SWITCH("is_stable", source1.is_stable() ? 1. : 0.));
      }

      if (include_features & FEATURE_MODULE_NON_ZERO) {
        FEATURIZE_DISPATCH_COND(FeaturizeScatterDimensionNumbers,
                                include_features, "scatter_dim_numbers",
                                source1.scatter_dimension_numbers());
        FEATURIZE_DISPATCH_COND(FeaturizeGatherDimensionNumbers,
                                include_features, "gather_dim_numbers",
                                source1.gather_dimension_numbers());
        FEATURIZE_DISPATCH(FixedNumericSequenceFeatures<6>,
                           "gather_slice_sizes", source1.gather_slice_sizes());
        out->push_back(FEATURIZE_SWITCH(
            "indices_are_sorted", source1.indices_are_sorted() ? 1. : 0.));
      }

      if (include_features & FEATURE_OP_ZERO) {
        out->push_back(FEATURIZE_SWITCH(
            "tuple_index", static_cast<float>(source1.tuple_index())));
        out->push_back(FEATURIZE_SWITCH(
            "exponent_bits", static_cast<float>(source1.exponent_bits())));
        out->push_back(FEATURIZE_SWITCH(
            "mantissa_bits", static_cast<float>(source1.mantissa_bits())));
        FEATURIZE_DISPATCH_COND(FeaturizeDotDimensionNumbers, include_features,
                                "dot_dim_numbers",
                                source1.dot_dimension_numbers());
        FEATURIZE_DISPATCH_COND(FeaturizeFftType, include_features, "fft_type",
                                source1.fft_type());
        FEATURIZE_DISPATCH(FixedNumericSequenceFeatures<2>, "fft_length",
                           source1.fft_length());
        // TODO: May include `.sharding()`
        out->push_back(FEATURIZE_SWITCH("is_host_transfer",
                                        source1.is_host_transfer() ? 1. : 0.));
        // TODO: May include `.domain_entry_sharding()`
        // TODO: May include `.domain_exit_sharding()`
        out->push_back(FEATURIZE_SWITCH("constrain_layout",
                                        source1.constrain_layout() ? 1. : 0.));
        // TODO: May include `.triangular_solve_options()`
        // TODO: May include `.cholesky_options()`
        // TODO: May include `.parameter_replication()`
        out->push_back(
            FEATURIZE_SWITCH("custom_call_has_side_effect",
                             source1.custom_call_has_side_effect() ? 1. : 0.));
        out->push_back(FEATURIZE_SWITCH("unique_indices",
                                        source1.unique_indices() ? 1. : 0.));
      }

      FEATURIZE_DISPATCH_COND(FeaturizeLayoutProto, include_features, "layout",
                              source1.shape().layout());
    })

}  // namespace ml_lib
}  // namespace xla

#endif  // THIRD_PARTY_PY_TPU_GRAPHS_PROCESS_DATA_XLA_FEATURIZERS_H_
