"""A simple script for inspect checkpoint files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
 
import sys
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("file_name", "", "Checkpoint filename")
tf.app.flags.DEFINE_string("tensor_name", "", "Name of the tensor to inspect")
def print_tensors_in_pb(file_name, tensor_name):
  try:
    graph_def = tf.GraphDef.FromString(open(file_name, 'rb').read())
    _inception_graph = tf.Graph()
    with _inception_graph.as_default():
      _ = tf.import_graph_def(graph_def, name='')
      session = tf.Session()
      for op in tf.get_default_graph().get_operations():
        print("op_name=",str(op.name), op)
      for key in tf.get_default_graph().get_all_collection_keys():
        print("collection_key=", key)
      #for values in tf.get_default_graph().get_collections():
      #  for value in values:
      #    print("var=", value)
  except Exception as e:  # pylint: disable=broad-except
    print(str(e))
    if "corrupted compressed block contents" in str(e):
      print("It's likely that your checkpoint file has been compressed ""with SNAPPY.")

def main(unused_argv):
  if not FLAGS.file_name:
    print("Usage: inspect_checkpoint --file_name=checkpoint_file_name "
    "[--tensor_name=tensor_to_print]")
    sys.exit(1)
  else:
    print_tensors_in_pb(FLAGS.file_name, FLAGS.tensor_name)
if __name__ == "__main__":
  tf.app.run()
