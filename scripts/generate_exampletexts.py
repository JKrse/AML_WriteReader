# ============================================================================================================

import tensorflow as tf
from config import *
import argparse
import os

config = Config()

# ============================================================================================================
model_architecture = "mlp_1_img_1_512_0"
local_files = config.local_path_temp
name = "proposals"
# ============================================================================================================

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument(
        "--model_architecture", type=str, default="mlp_1_img_1_512_0",
        help='Path to the file that contains the dataset')
parser.add_argument(
        "--name", type=str, default="proposals",
        help='Name of the method used')

args = parser.parse_args()

model_architecture = args.model_architecture
local_files = config.local_path
name = args.proposals
epochs = config.max_epoch


# ============================================================================================================

path = local_files / f"{name}_scoring"

# The network graph defined in .meta 
if not os.path.exists(f"{path}/{model_architecture}-{epochs}.meta"):
    print(".meta naming wrong")

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(f"{path}/{model_architecture}-{epochs}.meta")
    saver.restore(sess, tf.train.latest_checkpoint(path))

    graph = tf.get_default_graph()
    init_op = tf.global_variables_initializer()

    nodes = ([node.name for node in graph.as_graph_def().node])

    feed_dict = {graph.get_operation_by_name(nodes[0]).values(): x,
                 graph.get_operation_by_name(nodes[1]).values(): y_,
                 graph.get_operation_by_name(nodes[2]).values(): img,
                 graph.get_operation_by_name(nodes[3]).values(): 0.0008}

    print(sess.run(init_op, feed_dict))


#graph.get_operation_by_name(nodes[0]).values()

#fetches = graph.get_collection("train_op")

#data_train = np.load(os.path.join(local_path, f"data/proposals/data_train_full_3004.npy"), allow_pickle=True).item()