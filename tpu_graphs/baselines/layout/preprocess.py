import os
import numpy as np
import tqdm
import hashlib
import tensorflow as tf

from data import NpzDatasetPartition

class DatasetNormalizer():
  def __init__(self):
    self.fea_min = [np.float('inf'), np.float('inf'), np.float('inf')]
    self.fea_max =  [np.float('-inf'), np.float('-inf'), np.float('-inf')]
    self.used_columns = None


  def update_normalizer(self, feature_matrix):
    max_feat = np.max(feature_matrix, axis=0, keepdims=True) # 每列的最大值
    min_feat = np.min(feature_matrix, axis=0, keepdims=True) # 每列的最小值

    self.fea_min = np.min([self.fea_min, min_feat], axis=0, keepdims=True)
    self.fea_max = np.min([self.fea_max, max_feat], axis=0, keepdims=True)
    self.used_columns = self.fea_min != self.fea_max

  def apply_normalizer(self, feature_matrix, used_columns, min_feat, max_feat):
    used_columns = np.stack([used_columns for _ in feature_matrix.shape[0]], axis = 0)
    feature_matrix = feature_matrix[used_columns]
    min_feat = min_feat[used_columns]
    max_feat = max_feat[used_columns]
    return (feature_matrix - min_feat) / (max_feat - min_feat)


  def normalize(self, feature_matrix):
    return self.apply_normalizer(
          feature_matrix, self.used_columns, self.fea_min, self.fea_max
        )

  def save_normalizer(self, output):
     with open(output, 'w') as f:
        f.write(f"{self.fea_min},{self.fea_max},{self.used_columns}")

  def set_normalizer(self, min_fea, max_fea, used_columns):
     self.fea_max = max_fea
     self.fea_min = min_fea
     self.used_columns = used_columns
