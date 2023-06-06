import numpy as np
import tensorflow as tf
import tensorflow_gnn as tfgnn
from tpu_graphs.baselines.tiles import implicit


ADJ = np.array([[0, 1, 0],
                [0, 0, 1],
                [1, 1, 0]])
NONZERO = ADJ.nonzero()
TGT = NONZERO[0]  # adj[i, j] == 1  <==> i->j.
SRC = NONZERO[1]

GRAPH = tfgnn.GraphTensor.from_pieces(
    node_sets={
        'nodes': tfgnn.NodeSet.from_fields(
            sizes=tf.constant([3]), features={})},
    edge_sets={
        'edges': tfgnn.EdgeSet.from_fields(
            sizes=SRC.shape,
            features={},
            adjacency=tfgnn.Adjacency.from_indices(
                source=('nodes', SRC),
                target=('nodes', TGT)))})

AM = implicit.AdjacencyMultiplier(GRAPH, 'edges')

NP_RND = np.array(np.random.uniform(size=[3, 5, 2], low=-1, high=1), 'float32')
TF_RND = tf.constant(NP_RND, dtype=tf.float32)


class AdjacencyMultiplierTest(tf.test.TestCase):

  def test_colsums(self):
    self.assertAllEqual(AM.colsums(), ADJ.sum(0))

  def test_rowsums(self):
    self.assertAllEqual(AM.rowsums(), ADJ.sum(1))

  def test_matmul(self):
    result = AM @ TF_RND
    expected_result0 = ADJ.dot(NP_RND[..., 0])
    expected_result1 = ADJ.dot(NP_RND[..., 1])
    expected_result = np.stack([expected_result0, expected_result1], -1)
    self.assertAllClose(expected_result, result)

  def test_rmatmul(self):
    result = tf.transpose(TF_RND) @ AM
    expected_result0 = NP_RND.T[0].dot(ADJ)
    expected_result1 = NP_RND.T[1].dot(ADJ)
    expected_result = np.stack([expected_result0, expected_result1], 0)
    self.assertAllClose(expected_result, result)

  def test_add_eye(self):
    rnd = NP_RND[:, :, 0]
    self.assertAllClose(AM.add_eye() @ rnd, (ADJ + np.eye(3)).dot(rnd))
    self.assertAllClose(
        tf.transpose(rnd) @ AM.add_eye(), rnd.T.dot(ADJ + np.eye(3)))

  def test_normalize_right(self):
    rnd = NP_RND[:, :, 0]
    tf_rnd = TF_RND[:, :, 0]
    anorm_right = ADJ / ADJ.sum(1, keepdims=True)
    right_stochastic = AM.normalize_right()
    self.assertAllEqual(right_stochastic.rowsums(), tf.ones([3]))
    self.assertAllClose(anorm_right.dot(rnd), right_stochastic @ tf_rnd)
    self.assertAllClose(rnd.T.dot(anorm_right),
                        tf.transpose(tf_rnd) @ right_stochastic)

  def test_normalize_left(self):
    rnd = NP_RND[:, :, 0]
    tf_rnd = TF_RND[:, :, 0]
    anorm_left = ADJ / ADJ.sum(0, keepdims=True)
    left_stochastic = AM.normalize_left()
    self.assertAllEqual(left_stochastic.colsums(), tf.ones([3]))
    self.assertAllClose(anorm_left.dot(rnd), left_stochastic @ tf_rnd)
    self.assertAllClose(rnd.T.dot(anorm_left),
                        tf.transpose(tf_rnd) @ left_stochastic)

  def test_normalize_leftright(self):
    rnd = NP_RND[:, :, 0]
    tf_rnd = TF_RND[:, :, 0]

    anorm_leftright = ((ADJ / np.sqrt(ADJ.sum(1, keepdims=True)))
                       / np.sqrt(ADJ.sum(0, keepdims=True)))
    normalized_leftright = AM.normalize_leftright()
    self.assertAllClose(anorm_leftright.dot(rnd), normalized_leftright @ tf_rnd)
    self.assertAllClose(rnd.T.dot(anorm_leftright),
                        tf.transpose(tf_rnd) @ normalized_leftright)


if __name__ == '__main__':
  tf.test.main()
