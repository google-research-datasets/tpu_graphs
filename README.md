# TpuGraphs: A Performance Prediction Dataset on Large Tensor Computational Graphs

TpuGraphs is a performance prediction dataset on full tensor programs, represented as computational graphs, running on Tensor Processing Units (TPUs). Each graph in the dataset represents the main computation of a machine learning workload, e.g., a training epoch or an inference step. Each data sample contains a computational graph, a compilation configuration, and the execution time of the graph when compiled with the configuration. The graphs in the dataset are collected from open-source machine learning programs, featuring popular model architectures (e.g., ResNet, EfficientNet, Mask R-CNN, and Transformer).

*This is not an officially supported Google product.*

## Dataset

The dataset consists of two compiler optimization collections: *layout* and *tile*.
Layout configurations control how tensors are laid out in the physical memory, by specifying
the dimension order of each input and output of an operation node. A tile configuration controls
the tile size of each fused subgraph.

The dataset is located at http://download.tensorflow.org/data/tpu_graphs/v0.
You can use `wget` or `curl` command to download files.

- License: http://download.tensorflow.org/data/tpu_graphs/v0/LICENSE
- The statistics of all data collections can be found at http://download.tensorflow.org/data/tpu_graphs/v0/stat/*.csv. Please refer to http://download.tensorflow.org/data/tpu_graphs/v0/stat/README.md on the description of the statistics.
- Each tile data file is named as followed: http://download.tensorflow.org/data/tpu_graphs/v0/npz_tile_xla_{split}.tar
- Each layout data file is named as followed: http://download.tensorflow.org/data/tpu_graphs/v0/npz_layout_{source}_{search}_{split}.tar
  - {source}: `xla` or `nlp`
  - {search}: `default` or `random`
  - {split}: `train`, `valid`, or `test`

For example, to copy data for the layout:xla:random collection, run:

```sh
mkdir -p ~/data/tpugraphs
cd ~/data/tpugraphs

curl http://download.tensorflow.org/data/tpu_graphs/v0/npz_layout_xla_random_train.tar > npz_layout_xla_random_train.tar
curl http://download.tensorflow.org/data/tpu_graphs/v0/npz_layout_xla_random_valid.tar > npz_layout_xla_random_valid.tar
curl http://download.tensorflow.org/data/tpu_graphs/v0/npz_layout_xla_random_test.tar > npz_layout_xla_random_test.tar
tar xvf npz_layout_xla_random_train.tar
tar xvf npz_layout_xla_random_valid.tar
tar xvf npz_layout_xla_random_test.tar
```

To download all files, you may run (from a clone of this directory):

```sh
python3 echo_download_commands.py | bash
```

Removing the last pipe (`| bash`) shows the commands for downloading the dataset
(a few `curl` commands followed by `tar xvf`).


## Running Baseline Models

### Tile Size Model

#### Python environment setup with Conda

```sh
conda create -n tpugraphs python=3.10
conda activate tpugraphs

conda install -c conda-forge tensorflow=2.12
conda install -c conda-forge tqdm

pip install tensorflow_gnn==0.5.0
pip install tensorflow-ranking==0.5.2
conda clean --all
```

For subsequent runs, simply activate the same environment with `conda activate tpugraphs`.

#### Copy dataset files

To run the tiles baselines, you should download the tiles dataset:

```sh
# Create dataset directory
mkdir -p ~/data/tpugraphs
cd ~/data/tpugraphs
wget http://download.tensorflow.org/data/tpu_graphs/v0/LICENSE > LICENSE

# Copy data from to local dir.
cd ~/data/tpugraphs
curl http://download.tensorflow.org/data/tpu_graphs/v0/npz_tile_xla_train.tar > npz_tile_xla_train.tar
curl http://download.tensorflow.org/data/tpu_graphs/v0/npz_tile_xla_valid.tar > npz_tile_xla_valid.tar
curl http://download.tensorflow.org/data/tpu_graphs/v0/npz_tile_xla_test.tar > npz_tile_xla_test.tar
tar xvf npz_tile_xla_train.tar
tar xvf npz_tile_xla_valid.tar
tar xvf npz_tile_xla_test.tar
```

For a description of these files, please scroll towards the end of this page
("Dataset File Description").

#### Train model

The following command will train a GraphSAGE model with the early join of config features on a small subset of data:
```
python tiles_train.py --model=EarlyJoinSAGE --toy_data=True
```

To train on the full dataset, run:
```
python tiles_train.py --model=EarlyJoinSAGE
```

Once the training is done, it will produce a jsonz file with the prefix "run_".
This file will contain the overall top-K errors on kernels in the validation set.
To view the result:
```
zcat run_xxx.jsonz > run_xxx.json
```

Search for:
```
"final_error": {"val": {"1": <top-1 error>, "5": <top-5 error>, "10": <top-10 error>}}
```
where 0.2 error means 20% error.

Please refer to
[train_args.py](https://github.com/google-research-datasets/tpu_graphs/blob/main/baselines/tiles/train_args.py)
for a list of flags.

#### Sweep hyperparameters

Run Apache Beam locally (for debugging):
```
python tiles_beam_experiments.py --debug
```

To run the pipeline on Google Cloud, please follow [this instruction](https://cloud.google.com/dataflow/docs/quickstarts/create-pipeline-python).


#### Evaluate model
Once the training is done, the training output directory specified with
`--out_dir` (~/out/tpugraphs_tiles by default) will contain a model directory,
whose name starts with the prefix `model_`.

To evaluate a model(s), run:
```
python tiles_evaluate.py --dirs <comma-separated list of model dirs>
```

This script will print out per-program top-K errors for kernels in the validation set in the following format:
```
{
  "K": {  # top-K error
    <program> : <error>,
    ...
  },
  ...
}
```

## Layout Model

Instructions to train the baseline models for the layout collection can be found at https://github.com/kaidic/GST.

## Dataset File Description

### Tiles Collection `.npz` files

We provide our dataset as `.npz` files. Download instructions are in Section
"Copy dataset files".

Suppose a `.npz` file stores a graph (representing a kernel) with `n` nodes and
`m` edges. In addition, suppose we compile the graph with `c` different
configurations, and run each on a TPU. **Crucially, the configuration is at the
graph-level**. Then, the `.npz` file stores the following dictionary (can be
loaded with `d = dict(np.load("npz/tile/xla/train/<pick 1>.npz"))`):

  * Key `"node_feat"`: contains `float32` matrix with shape `(n, 140)`. The
    `u`th row contains the feature vector for node `u < n` (please see
    Subsection "Node Features", below).
    Nodes are ordered topologically.
  * Key `"node_opcode"` contains `int32` vector with shape `(n, )`. The `u`th
    entry stores the op-code for node `u` (please see the mapping of opcode to
    instruction name
    [here](https://github.com/google-research-datasets/tpu_graphs/blob/main/tpu_graphs/process_data/xla/hlo_opcode.h#L94)).
  * Key `"edge_index"` contains `int32` matrix with shape `(m, 2)`. If entry
    `i` is `= [u, v]`  (where `0 <= u, v < n`), then there is a directed edge
    from node `u` to node `v`, where `u` consumes the output of `v`.
  * Key `"config_feat"` contains `float32` matrix with shape `(c, 24)` with row
    `j` containing the (graph-level) configuration feature vector (please see
    Subsection "Tile Config Features").
  * Keys `"config_runtime"` and `"config_runtime_normalizers"`: both are `int64`
    vectors of length `c`. Entry `j` stores the runtime (in nanoseconds) of the
    given graph compiled with configuration `j` and a default configuration,
    respectively. Samples from the same graph may have slightly different `"config_runtime_normalizers"` because they are measured from different runs
    on multiple machines.

Finally, for the tile collection, your job is to predict the indices of the best
configurations (i.e., ones leading to the smallest `d["config_runtime"] / d["config_runtime_normalizers"]`).

### Layout Collections `.npz` files

Suppose a `.npz` file stores a graph (representing the entire program) with `n`
nodes and `m` edges. In addition, suppose we compile the graph with `c`
different configurations, and run each on a TPU. **Crucially, the configuration
is at the node-level.**. Suppose that `nc` of the `n` nodes are configurable.
Then, the `.npz` file stores the following dictionary (can be loaded with, e.g.,
`d = dict(np.load("npz/layout/xla/random/train/unet3d.npz"))`):

  * Keys `"node_feat"`, `"node_opcode"`, `"edge_index"`,  are like above.
  * Key `"node_config_ids"` contains `int32` vector with shape `(nc, )` and
    every entry is in `{0, 1, ..., n - 1}` i.e. indicating the indices of the
    configurable nodes. For these nodes, they can have an additional feature
    vector that instructs the compiler (described next).
  * Key `"node_config_feat"` contains `float32` tensor  with shape
    `(c, nc, 18)`. Entry `[j, k]` gives an 18-dimensional vector describing the
    configuration features for node `d["node_config_ids"][k]` for the `j`th run
    (please see Subsection "Layout Config Features", below).
  * Key `"config_runtime"` contains `int32` vector with shape `(c, )` where the
    `j`th entry contains the runtime of the `j`th run (i.e., when nodes are
    configured with `d["node_config_feat"][j]`).

Finally, for the layout collections, your job is to predict sort the indices
from best-to-worse configurations (i.e., ones leading to the smallest
`d["config_runtime"]`). We do not have to use runtime normalizers for this task
because the runtime variation at the entire program level is very small.

Optionally, you may access key `"node_splits"`, which is a variable-length list
of node IDs that are the starting of HLO computations in the graph (similar to
functions in a program). Essentially, nodes `d["node_splits"][i]` to
`d["node_splits"][i+1] - 1` belongs to the same computation. If you want to
partition the graph into multiple segments, this information may be useful,
e.g., putting nodes from the same computation in the same partition. However,
you may compute your own partitioning (e.g., using METIS) as well.


## Features

### Node Features

To extract a node feature vector, we either copy values from various fields in an XLA’s HLO instruction (a node in an HLO graph) as they are, or convert categorical values using one-hot encoding. To convert an unbounded list of numbers (e.g. tensor shape) to a fixed-size vector, we truncate the list to six elements and include the summation and/or product of all elements in the list (e.g., the product of dimension sizes represents the volume of the tensor). In our dataset, none of the tensors has more than six dimensions. The code for node features extraction can be found [here](https://github.com/google-research-datasets/tpu_graphs/blob/main/tpu_graphs/process_data/xla/featurizers.h#L542).

The following describe each element at a particular index in the node feature vector.

```
0: is_root - whether this node is the output
1: element_size_in_bits - deprecated, always 0
// 2–20: One hot vector of shape_element_type.
2: shape_element_type_is_invalid_type
3: shape_element_type_is_pred
4: shape_element_type_is_s8
5: shape_element_type_is_s16
6: shape_element_type_is_s32
7: shape_element_type_is_s64
8: shape_element_type_is_u8
9: shape_element_type_is_u16
10: shape_element_type_is_u32
11: shape_element_type_is_u64
12: shape_element_type_is_f16
13: shape_element_type_is_f32
14: shape_element_type_is_f64
15: shape_element_type_is_bf16
16: shape_element_type_is_c64
17: shape_element_type_is_c128
18: shape_element_type_is_tuple
19: shape_element_type_is_opaque_type
20: shape_element_type_is_token
// 21–28: Size (number of elements) for each dimension, or an upper bound on the size if the dimension is dynamic.  In XLA, dimensions are numbered from 0 to N-1 for an N-dimensional array. The first element of 'shape_dimensions' is the size of dimension 0, the second element is the size of dimension 1, and so forth.  Empty list indicates a scalar.
21: shape_dimensions_0
22: shape_dimensions_1
23: shape_dimensions_2
24: shape_dimensions_3
25: shape_dimensions_4
26: shape_dimensions_5
27: shape_dimensions_sum
28: shape_dimensions_product
29: shape_tuple_shapes_size - for tuples only, the shapes of constituent shapes in the tuple sequence
30: parameter_number = K - indicating that is is the Kth parameter to the computation, only for Parameter operation
// 31–36: Dimensions present for some operations that require reshaping or broadcasting, including Reshape, Reduce, ReduceWindow, and Reverse.
31: dimensions_0
32: dimensions_1
33: dimensions_2
34: dimensions_3
35: dimensions_4
36: dimensions_5
// 37–92: Windowing information in an operation such as convolution. The window is moved across a base area and for each position of the window a computation is performed.
37: window_size_0
38: window_size_1
39: window_size_2
40: window_size_3
41: window_size_4
42: window_size_5
43: window_size_sum
44: window_size_product
45: window_stride_0
46: window_stride_1
47: window_stride_2
48: window_stride_3
49: window_stride_4
50: window_stride_5
51: window_stride_sum
52: window_stride_product
53: window_padding_low_0
54: window_padding_low_1
55: window_padding_low_2
56: window_padding_low_3
57: window_padding_low_4
58: window_padding_low_5
59: window_padding_low_sum
60: window_padding_low_product
61: window_padding_high_0
62: window_padding_high_1
63: window_padding_high_2
64: window_padding_high_3
65: window_padding_high_4
66: window_padding_high_5
67: window_padding_high_sum
68: window_padding_high_product
// 69–76: Dilation factor of the sliding window. A dilation factor of 1 means no dilation. window_dilation - 1 no-op entries ("holes") are implicitly placed between each kernel element.
69: window_window_dilation_0
70: window_window_dilation_1
71: window_window_dilation_2
72: window_window_dilation_3
73: window_window_dilation_4
74: window_window_dilation_5
75: window_window_dilation_sum
76: window_window_dilation_product
// 77-84: Dilation factor of the base area. A dilation factor of 1 means no dilation. base_dilation - 1 no-op entries ("holes") are implicitly placed between each base area element.
77: window_base_dilation_0
78: window_base_dilation_1
79: window_base_dilation_2
80: window_base_dilation_3
81: window_base_dilation_4
82: window_base_dilation_5
83: window_base_dilation_sum
84: window_base_dilation_product
// 85-92: Window reversal means that this dimension was logically reversed before the operation.
85: window_window_reversal_0
86: window_window_reversal_1
87: window_window_reversal_2
88: window_window_reversal_3
89: window_window_reversal_4
90: window_window_reversal_5
91: window_window_reversal_true_count
92: window_window_reversal_false_count
// 93–106: The dimension numbers used for a convolution.
93: convolution_dim_numbers_input_batch_dim - the dimension number that represents batch in the input
94: convolution_dim_numbers_input_feature_dim - the dimension number that represents features in the input
// 95–98: Dimension numbers for the spatial dimensions that the window moves through in the input.
95: convolution_dim_numbers_input_spatial_dims_0
96: convolution_dim_numbers_input_spatial_dims_1
97: convolution_dim_numbers_input_spatial_dims_2
98: convolution_dim_numbers_input_spatial_dims_3
99: convolution_dim_numbers_kernel_input_feature_dim - the dimension number that represents input features in the convolutional kernel (rhs)
100: convolution_dim_numbers_kernel_output_feature_dim - the dimension number that represents output features in the convolutional kernel (rhs)
// 101-104: Dimension numbers for the spatial dimensions that the window moves through in the kernel (rhs). window.strides(0) is the stride in the kernel_spatial_dimensions(0) dimension.
101: convolution_dim_numbers_kernel_spatial_dims_0
102: convolution_dim_numbers_kernel_spatial_dims_1
103: convolution_dim_numbers_kernel_spatial_dims_2
104: convolution_dim_numbers_kernel_spatial_dims_3
105: convolution_dim_numbers_output_batch_dim - the dimension number that represents batch in the output
106: convolution_dim_numbers_output_feature_dim - the dimension number that represents features in the output
107: feature_group_count - the number of feature groups, used for a convolution. Must be a divisor of the input feature dimension and output feature dimension. If not specified, it will use a default value of 1.
108: batch_group_count - the number of batch groups, used for a convolution.
// 109–120: [begin/start, end/limit) index range and stride for a slice operation.
109: slice_dims_start_0
110: slice_dims_start_1
111: slice_dims_start_sum
112: slice_dims_start_product
113: slice_dims_stride_0
114: slice_dims_stride_1
115: slice_dims_stride_sum
116: slice_dims_stride_product
117: slice_dims_limit_0
118: slice_dims_limit_1
119: slice_dims_limit_sum
120: slice_dims_limit_product
// 121 - 124: [start, start + size) range size for a dynamic slice ('start' is specified dynamically in the second operand of the operation).
121: dynamic_slice_sizes_0
122: dynamic_slice_sizes_1
123: dynamic_slice_sizes_sum
124: dynamic_slice_sizes_product
// 125–132: Padding configuration that describes the edge padding of a pad operation.
125: padding_config_edge_padding_low_0
126: padding_config_edge_padding_low_1
127: padding_config_edge_padding_low_sum
128: padding_config_edge_padding_low_product
129: padding_config_edge_padding_high_0
130: padding_config_edge_padding_high_1
131: padding_config_edge_padding_high_sum
132: padding_config_edge_padding_high_product
133: is_stable - whether this Sort operation should be stable
// 134–139: Physical layout used to pack the tensor shape.
134: layout_minor_to_major_0
135: layout_minor_to_major_1
136: layout_minor_to_major_2
137: layout_minor_to_major_3
138: layout_minor_to_major_4
139: layout_minor_to_major_5
```

Suffix _i, where i is an integer, indicates the information for the tensor dimension i. If a tensor has N dimensions, feature values of _i are set to 0 if i >= N (0 padding). Suffix _sum is the summation of the feature values across all dimensions. Suffix _product is the product of the feature values across all dimensions.

The source code of the feature extractor can be found [here](https://github.com/google-research-datasets/tpu_graphs/blob/main/tpu_graphs/process_data/xla/featurizers.h#L542), which extracts features/attributes from HloProto defined [here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/hlo.proto).

### Tile Config Features

The following describe each element at a particular index in the tile config feature vector.

```
// 0–7: Tile sizes of the convolution kernel, only for a convolution operation.
0: kernel_bounds_0
1: kernel_bounds_1
2: kernel_bounds_2
3: kernel_bounds_3
4: kernel_bounds_4
5: kernel_bounds_5
6: kernel_bounds_sum
7: kernel_bounds_product
// 8–15: Output tile sizes.
8: output_bounds_0
9: output_bounds_1
10: output_bounds_2
11: output_bounds_3
12: output_bounds_4
13: output_bounds_5
14: output_bounds_sum
15: output_bounds_product
// 16-23: Input tile sizes.
16: input_bounds_0
17: input_bounds_1
18: input_bounds_2
19: input_bounds_3
20: input_bounds_4
21: input_bounds_5
22: input_bounds_sum
23: input_bounds_product
```

Note that input_bounds are usually set to 0 because they can be inferred by the compiler from output_bounds (and kernel_bounds). If a tensor has N dimensions, feature values of _i are set to 0 if i >= N (0 padding).

### Layout Config Features

The following describe each element at a particular index in the per-node layout config feature vector.

```
// 0–5: Physical layout of the output tensor
0: output_layout_0
1: output_layout_1
2: output_layout_2
3: output_layout_3
4: output_layout_4
5: output_layout_5
// 6-11: Physical layout of the input  tensor
6: intput_layout_0
7: intput_layout_1
8: intput_layout_2
9: intput_layout_3
10: intput_layout_4
11: intput_layout_5
// 12-17: Physical layout of the kernel tensor, only for a convolution operation
12: kernel_layout_0
13: kernel_layout_1
14: kernel_layout_2
15: kernel_layout_3
16: kernel_layout_4
17: kernel_layout_5
```

If a tensor has N dimensions, feature values of _i are set to -1 if i >= N (-1 padding). A layout determines the order of minor-to-major tensor dimensions. For example, the layout of {1, 0, 2, -1, -1, -1} of a 3D tensor indicates that dimension 1 is the most minor (elements of the most minor dimension are consecutive in the physical space) and dimension 2 is the most major.
