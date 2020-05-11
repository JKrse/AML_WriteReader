from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import os
import random
import pandas as pd
import copy

from tensorflow.python.util import nest
from discriminator import *
from config import *

tf.app.flags.DEFINE_string('data_path', f"{Config.local_path}/data",
                           """Path where the data will be loaded.""")
tf.app.flags.DEFINE_string('name', "neuraltalk", #"neuraltalk" "proposals", # skal lige hardcodes om (sat til neuraltalk for nemhedens skyld)
                           """Path where the data will be loaded.""")
tf.app.flags.DEFINE_string('model_architecture', 'mlp_1_img_1_512_0',
                           """Number of images to process in a batch.""")
tf.app.flags.DEFINE_integer('epochs', Config.max_epoch, # Change - hardcoded 30
                           """Number of epochs.""")
args = tf.app.flags.FLAGS

def main(_):
    exp_name = "%s_scoring"%(args.name)
    local_files = Config.local_path
    log_path = local_files / exp_name
    save_path = local_files / exp_name
    data_path = os.path.join(args.data_path, args.name)
    save_path_train = f"{local_files}/{exp_name}/train"
    save_path_test = f"{local_files}/{exp_name}/test"

    if not os.path.exists(local_files):
        os.makedirs(local_files)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path_train):
        os.mkdir(save_path_train)
    if not os.path.exists(save_path_test):
        os.mkdir(save_path_test)

    train_models = [args.name]
    test_models = [args.name, 'human']

    config = Config()
    config = config_model_coco(config, args.model_architecture)
    config.max_epoch = args.epochs

    if config.random_search:
        # Parameters investigated using Random Search
        config.learning_rate = np.round(random.expovariate(100000),10)
        config.num_layers = random.randint(1,2,3)
        config.dropout_prob = np.round(random.uniform(0.05, 0.5),2)
        config.vocab_size = random.choice([3004, 5004, 10004])
    
    if args.name == "neuraltalk":
        load_features = True
    else:
        load_features = False

    [data_train, data_val, data_test, word_embedding] = data_loader(
            data_path, use_mc_samples=False, load_features=load_features)
    word_to_idx = data_train[f'word_to_idx']

    # if config.resize_data:
    #     data_train = resize_data(data_train, config.resize_samples)
    if not config.resize_data:
        config.resize_samples = len(data_train["file_names"]) 

    print("Model architecture:%s"%(args.model_architecture))
    with tf.Graph().as_default():
        with tf.name_scope("Train"):
            with tf.variable_scope("Discriminator", reuse=None):
                mtrain = Discriminator(word_embedding, word_to_idx, use_glove=True,
                                       config=config, is_training=True)
            tf.summary.scalar("Training Loss", mtrain._loss)
            tf.summary.scalar("Training Accuracy", mtrain._accuracy)

        with tf.name_scope("Val"):
            with tf.variable_scope("Discriminator", reuse=True):
                mval = Discriminator(word_embedding, word_to_idx, use_glove=True,
                                     config=config, is_training=False)
            tf.summary.scalar("Validation Loss", mval._loss)
            tf.summary.scalar("Validation Accuracy", mval._accuracy)

        config_sess = tf.ConfigProto(allow_soft_placement=True)
        config_sess.gpu_options.allow_growth = True
        with tf.Session(config=config_sess) as sess:
            tf.global_variables_initializer().run()
            summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver()
            
            # model_architecture / num_layers / dropout_prob / batch_size / use_lstm
            output_filename = f"vocab{config.vocab_size}__model_{args.model_architecture}__lr{config.learning_rate}__" \
                f"lay{config.num_layers}__dp{config.dropout_prob}__bs{config.batch_size}__lstm{config.use_lstm}" \
                f"__ts{config.resize_samples}.txt"

            output_filename_train = f"vocab{config.vocab_size}__model_{args.model_architecture}__lr{config.learning_rate}__" \
                f"lay{config.num_layers}__dp{config.dropout_prob}__bs{config.batch_size}__lstm{config.use_lstm}" \
                f"__ts{config.resize_samples}_train.txt"

            output_filename_test = f"vocab{config.vocab_size}__model_{args.model_architecture}__lr{config.learning_rate}__" \
                f"lay{config.num_layers}__dp{config.dropout_prob}__bs{config.batch_size}__lstm{config.use_lstm}" \
                f"__ts{config.resize_samples}_testresults.txt"

            output_filepath = os.path.join(save_path, output_filename)
            output_filepath_train = os.path.join(save_path_train, output_filename_train)
            output_filepath_test = os.path.join(save_path_test, output_filename_test)

            f = open(output_filepath, 'w')
            f_train = open(output_filepath_train, 'w')
            # Column names:
            f.write(f"{test_models[0]} average score\tacc {test_models[0]}\t" 
                    f"{test_models[1]} average score\tacc {test_models[1]}\n")

            f_train.write("Epoch\t Loss\t Accuracy\n")

            score_list = []
            idx_batch_list = []
            cat_list = []

            # Training
            for i in range(config.max_epoch):
                print(f"Epoch: {i + 1} out of {config.max_epoch}")
                train_loss, train_acc = train(sess, mtrain, data_train,
                                              gen_model=train_models, epoch=i,
                                              config=config)

                for k_item in range(len(train_loss)):
                    f_train.write(f"{i} \t")
                    f_train.write(f"{train_loss[k_item]} \t")
                    f_train.write(f"{train_acc[k_item]} \n")

                for test_model in test_models:
                    [acc, logits, scores, idx_batch] = inference(
                            sess, mval, data_val, test_model, config=config)
                    s = np.mean(scores[:,:,0])
                    f.write("%f\t" % s)
                    a = np.mean(acc) # Average Score: 
                    f.write("%f\t" % a)

                    if i == config.max_epoch - 1:

                        if test_model == "human":
                            cat_list += list(np.ones(len(idx_batch)))
                        elif test_model == args.name:
                            cat_list += list(np.zeros(len(idx_batch)))

                        score_list += scores[:,:,0].tolist()
                        idx_batch_list += idx_batch.tolist()

                f.write("\n")

            all_test = pd.DataFrame()
            all_test["idx_batch"] = idx_batch_list
            all_test["cat"] = cat_list
            all_test["score"] = score_list
            all_test.to_csv(output_filepath_test, sep="\t", header=True)

            f.close()
            f_train.close()



            if save_path:
                model_path = os.path.join(save_path, args.model_architecture)
                print("Saving model to %s." % model_path)
                saver.save(sess, model_path, global_step=i+1)
                print("Model saved to %s." % model_path)


if __name__ == "__main__":
    tf.app.run()
