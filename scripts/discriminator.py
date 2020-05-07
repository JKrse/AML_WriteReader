from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf
import os
import random
import copy

from tensorflow.python.util import nest
from config import *

import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0,os.path.join(dir_path, "tensorflow_compact_bilinear_pooling"))

class ResidualWrapper(tf.contrib.rnn.RNNCell):
  """RNNCell wrapper that ensures cell inputs are added to the outputs."""

  def __init__(self, cell):
    """Constructs a `ResidualWrapper` for `cell`.
    Args:
      cell: An instance of `RNNCell`.
    """
    self._cell = cell

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    """Run the cell and add its inputs to its outputs.
    Args:
      inputs: cell inputs.
      state: cell state.
      scope: optional cell scope.
    Returns:
      Tuple of cell outputs and new state.
    Raises:
      TypeError: If cell inputs and outputs have different structure (type).
      ValueError: If cell inputs and outputs have different structure (value).
    """
    outputs, new_state = self._cell(inputs, state, scope=scope)
    nest.assert_same_structure(inputs, outputs)
    # Ensure shapes match
    def assert_shape_match(inp, out):
      inp.get_shape().assert_is_compatible_with(out.get_shape())
    nest.map_structure(assert_shape_match, inputs, outputs)
    res_outputs = nest.map_structure(
        lambda inp, out: inp + out, inputs, outputs)
    return (res_outputs, new_state)

def add_mc_samples(data, mc_samples):
    file_names = data['file_names']
    name = list(mc_samples['captions']['dis'][file_names[0]])
    for file_name in file_names:
        ## Their method: 
        # data['captions']['dis'][file_name]['mc_samples'] = mc_samples[file_name]['gen'] # same example files
    
        # Change: 
        data['captions']['dis'][file_name]['mc_samples'] = mc_samples['captions']['dis'][file_name]["random_word_gen"] # This is kinda wierd.. 
        """
        Adding an extra dictionary entry: 'mc_samples'
        
        mc_samples from the dumped files, seem to have the structure: {"file_name": "caption"}. 

        We now generate mc_samples the exact same way as human data, meaning the structure is the same. 
        
        Thus, it reference to the same image but the generated captions. 
        """
    return data

def data_loader(data_path=None, data_type = '_full', use_mc_samples=False, load_features=False):
    """
    Data format (compatible with Show Attend and Tell):
    the data file is a dict has the following keys:
    'file_names'
    'image_idxs'
    'captions': a dict has keys 'gen' for generator and 'dis' for discriminator
    'features': a dict has keys 'gen' for generator and 'dis' for discriminator
                (to be loaded when needed)
    'word_to_idx': a dict with word to idx mapping
    """
    data_train = np.load(os.path.join(data_path, f"data_train_full_{Config.vocab_size}.npy"), allow_pickle=True).item()
    data_val = np.load(os.path.join(data_path, f"data_val_full_{Config.vocab_size}.npy"), allow_pickle=True).item()
    data_test = np.load(os.path.join(data_path, f"data_test_full_{Config.vocab_size}.npy"), allow_pickle=True).item()
    if use_mc_samples == True:
        mc_train = np.load(os.path.join(data_path, 'dumped_train.npy'), allow_pickle=True).item()
        mc_val = np.load(os.path.join(data_path, 'dumped_val.npy'), allow_pickle=True).item()
        mc_test = np.load(os.path.join(data_path, 'dumped_test.npy'), allow_pickle=True).item()
        data_train = add_mc_samples(data_train, mc_train)
        data_val = add_mc_samples(data_val, mc_val)
        data_test = add_mc_samples(data_test, mc_test)

    if load_features: 
        data_train['features']['dis'] = np.load(
                f'{Config.local_path}/data/resnet152/feature_dis_train%s.npy' % (data_type), encoding = 'latin1', allow_pickle=True).item()
        data_val['features']['dis'] = np.load(
                f'{Config.local_path}/data/resnet152/feature_dis_val%s.npy' % (data_type), encoding = 'latin1', allow_pickle=True).item()
        data_test['features']['dis'] = np.load(
                f'{Config.local_path}/data/resnet152/feature_dis_test%s.npy' % (data_type), encoding = 'latin1', allow_pickle=True).item()

    word_embedding = np.load(
            f'{Config.local_path}/data/word_embedding_{str(Config().embedding_size)}_{str(Config().vocab_size)}.npy', allow_pickle=True)

    return [data_train, data_val, data_test, word_embedding]


def resize_data(dict_data, data_size=None): 
    """
    Helper function to decrease samples used in an attempt to overfit
    
    dict_data 
        dictionary data
    data_size
        The number of samples wanted 
    """
    if len(dict_data["file_names"]) < data_size or not isinstance(data_size, int):
        print(f"Data has {len(dict_data['file_names'])}")
        data_size = len(dict_data["file_names"])
    
    data = {}
    data["file_names"] = dict_data["file_names"][0:data_size]
    data["image_idxs"] = dict_data["image_idxs"][0:data_size]
    data["word_to_idx"] = dict_data["word_to_idx"]
    data["captions"] = {}
    data["captions"]["gen"] = dict_data["captions"]["gen"][0:data_size]
    data["captions"]["dis"] = dict(list(dict_data["captions"]["dis"].items())[0:data_size])
    data["features"] = {}
    data["features"]["dis"] = dict(list(dict_data["features"]["dis"].items())[0:data_size])
    return data


class Discriminator(object):
    """The model."""
    def __init__(self, word_embedding, word_to_idx=None, use_glove=True,
                 is_training=True, dim_feat=2048, config=Config(), num_input=2):
        self.x        = tf.placeholder(tf.int32, [None, config.num_steps])
        self.y_       = tf.placeholder(tf.float32, [None, 2])
        self.img_feat = tf.placeholder(tf.float32, [None, dim_feat])
        self.lr       = tf.placeholder(tf.float32)
        self._eos = word_to_idx['<eos>']
        mask = tf.to_float(tf.equal(self.x, self._eos))

        num_steps = config.num_steps
        hidden_size = config.hidden_size
        vocab_size = config.vocab_size
        embedding_size = config.embedding_size
        num_input = config.num_input
        use_img_feat = config.use_img_feat
        use_lstm     = config.use_lstm
        combine_typ  = config.combine_typ
        cls_hidden   = config.cls_hidden
        use_residual = config.use_residual

        img_feat = tf.layers.dense(inputs=self.img_feat, units=hidden_size, activation=None)

        if use_residual:
            def lstm_cell():
                return ResidualWrapper(tf.contrib.rnn.BasicLSTMCell(
                    hidden_size, forget_bias=1.0, state_is_tuple=True))
        else:
            def lstm_cell():
                return tf.contrib.rnn.BasicLSTMCell(
                    hidden_size, forget_bias=1.0, state_is_tuple=True)

        attn_cell = lstm_cell
        if is_training and config.dropout_prob < 1:
            def attn_cell():
                return tf.contrib.rnn.DropoutWrapper(
                    lstm_cell(), output_keep_prob=config.dropout_prob)
        cell = tf.contrib.rnn.MultiRNNCell(
            [attn_cell() for _ in range(config.num_layers)], state_is_tuple=True)

        if use_glove:
            embedding = tf.get_variable(
                "embedding", dtype=tf.float32, initializer=tf.constant(word_embedding))
        else:
            embedding = tf.get_variable(
                "embedding", [vocab_size, embedding_size],
                initializer=tf.random_uniform_initializer(minval=-1.0, maxval=1.0))
        inputs = tf.nn.embedding_lookup(embedding, self.x)

        if use_img_feat == 'concat_bf_lstm':
            raise Exception("use_img_feat=concat_bf_lstm not supported")
            img_reshape = tf.reshape(img_feat, [-1, 1, dim_feat])
            img_tiled   = tf.tile(img_reshape, [1, num_steps, 1])
            inputs = tf.concat([inputs, img_tiled], 2)

        if is_training and config.dropout_prob < 1:
            inputs = tf.nn.dropout(inputs, config.dropout_prob)


        if use_lstm:
            ta_d_outputs = tf.TensorArray(
                dtype=tf.float32, size=num_steps,
                dynamic_size=False, infer_shape=True)

            state = cell.zero_state(tf.shape(inputs)[0], tf.float32)
            with tf.variable_scope("RNN"):
                for time_step in range(num_steps):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    (output, state) = cell(inputs[:, time_step, :], state)
                    ta_d_outputs = ta_d_outputs.write(time_step, output)

                # batch_size x seq_length x hidden_size
                ta_d_outputs = tf.transpose(
                    ta_d_outputs.stack(), perm=[1, 0, 2])

                # apply the mask
                mask = tf.expand_dims(mask, -1)
                mask = tf.tile(mask, tf.stack([1, 1, hidden_size]))
                masked_out = ta_d_outputs * mask
                output = tf.reduce_sum(masked_out, axis=1)
                output_context, output_candidate = tf.split(
                        output, num_or_size_splits=num_input, axis=0)
        else:
            inputs = tf.reshape(inputs, [-1, num_steps * embedding_size])
            output_context, output_candidate = tf.split(
                    inputs, num_or_size_splits=num_input, axis=0)

        print("-"*80)
        if use_img_feat == 'concat_af_lstm':
            print("Image feature concatenate after the contextfeature from LSTM")
            imgf_1, imgf_2 = tf.split(img_feat, num_or_size_splits=num_input, axis=0)
            output_context = tf.concat([imgf_1, output_context], axis=1)
        elif use_img_feat == 'only_img':
            print("Image Feature Replacing the Context Feature from LSTM")
            imgf_1, imgf_2 = tf.split(img_feat, num_or_size_splits=num_input, axis=0)
            output_context = imgf_1
        else:
            print("Not using image feature")
        print("-"*80)

        # Combining candidate information with context information
        print("-"*80)
        if combine_typ == 'concat':
            print("Directly concatenate context and candidate feature.")
            output = tf.concat([output_context, output_candidate], axis=1)
        elif combine_typ == 'bilinpool':    # compact bilinear
            from compact_bilinear_pooling import compact_bilinear_pooling_layer as compact_bilinear_pooling
            print("Use compact bilinear pooling between candidate/context features.")
            out_dim = 8192
            output_context   = tf.expand_dims(tf.expand_dims(output_context, 1), 1)
            output_candidate = tf.expand_dims(tf.expand_dims(output_candidate, 1), 1)
            output = compact_bilinear_pooling(output_context, output_candidate, out_dim)
            output = tf.reshape(output, [-1, out_dim]) # make static time shape
        else:
            print("Use only the candidate feature.")
            output = output_candidate
        print("-"*80)

        for _ in range(cls_hidden):
            output = tf.layers.dense(inputs=output, units=512, activation=tf.nn.relu)
            if is_training and config.dropout_prob < 1:
                output = tf.nn.dropout(output, config.dropout_prob)

        y = tf.layers.dense(inputs=output, units=2, activation=None)

        score = tf.nn.softmax(y, dim=-1, name=None)

        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=y))

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(self.y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self._logits = y
        self._score = score
        self._loss = loss
        self._accuracy = accuracy

        if not is_training:
            return

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), config.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())


def train(sess, model, data, gen_model, epoch, dim_feat=2048, config=Config(), verbose=True):
    """Runs the model on the given data."""
    start_time = time.time()
    # construct two pairs for each image: (real0, real1), (real0, fake)
    batch_size = int(config.batch_size / 2)
    num_steps = config.num_steps
    num_input = config.num_input

    filename = data['file_names']

    fetches = {
        "loss": model._loss,
        "accuracy": model._accuracy,
        "train_op": model._train_op
    }

    if len(gen_model) == 0:
        idx = list(range(len(filename)))
    else:
        idx = list(range(len(filename)*len(gen_model)))
    random.shuffle(idx)
    epoch_size = len(idx) // batch_size
    if batch_size * epoch_size < len(idx):
        epoch_size += 1
        idx.extend(idx[:batch_size * epoch_size - len(idx)])
    
    print(f"Epoch size (amount of batch steps for 1 epoch): {epoch_size}")

    negative_samples_idx = []
    pathological_transf_idx = []

    if len(gen_model) > 0:
        negative_samples_idx.append(0)

    if config.use_random_human or config.use_random_word or config.use_word_permutation:
        negative_samples_idx.append(2)
        if config.use_random_human:
            pathological_transf_idx.append(0)
        if config.use_random_word:
            pathological_transf_idx.append(1)
        if config.use_word_permutation:
            pathological_transf_idx.append(2)

    if config.use_mc_samples:
        negative_samples_idx.append(1)
    print("Negative Samples    : %s"%negative_samples_idx)
    print("Pathlogical Samples : %s"%pathological_transf_idx)

    mc_sample = True
    if not "mc_samples" in data['captions']['dis'][filename[0]]:
        mc_sample = False

    loss_total = []
    accuracy_total = []

    for i in range(epoch_size): 
    # for i in range(300): # CHANGE
        if i == epoch_size - 1:
            idx_batch = idx[batch_size*i:]
        else:
            idx_batch = idx[batch_size*i:batch_size*(i+1)]
        x   = np.zeros((len(idx_batch)*num_input*2, num_steps), dtype=np.int32)
        y_  = np.zeros((len(idx_batch)*2, 2), dtype=np.float32)
        img = np.zeros((len(idx_batch)*num_input*2, dim_feat))

        idx_batch = [ int(tmp_idx_b % len(filename)) for tmp_idx_b in idx_batch ]

        for j in range(len(idx_batch)):
            """
            Train the real caption against 3 types other caption types:
            if 0: 
                Machine generated captions (e.g. proposals [or Neuraltalk])
            if 1: 
                Machicne sample (our noicy data) **Not used here**
            if 2:
                Random Captions(RC)     [0]
                Random Word (RW)        [1]
                Word Permutation(WP)    [2]
            """
            curr_img = copy.deepcopy(data['features']['dis'][filename[idx_batch[j]]])
            real_cap = copy.deepcopy(data['captions']['dis'][filename[idx_batch[j]]]['human'])
            # 1st pair: (real0, real1)
            if len(real_cap.shape) > 1:
                real_idx = list(range(len(real_cap)))
                random.shuffle(real_idx)
                x[j*2, :]   = real_cap[real_idx[0]]
            else: 
                x[j*2, :]   = real_cap
            img[j*2,:]  = curr_img

            if len(real_cap.shape) > 1:
                x[j*2+len(idx_batch)*num_input, :]   = real_cap[real_idx[1]]
            else:
                x[j*2+len(idx_batch)*num_input, :] = real_cap
            
            img[j*2+len(idx_batch)*num_input, :] = curr_img
            y_[j*2, 0] = 1.0

            # 2nd pair: (real0, fake), fake is sampled from (gen, random_human, random_word)
            if len(real_cap.shape) > 1:
                x[j*2+1, :]  = real_cap[real_idx[0]]
            else: 
                x[j*2+1, :]  = real_cap
            
            img[j*2+1,:] = curr_img
            y_[j*2+1, 1] = 1.0

            if mc_sample: 
                rand_ind = np.random.choice(negative_samples_idx)
            else: 
                rand_ind = np.random.choice([0,2])

            if rand_ind == 0: # Use machine generated captions
                if type(gen_model) == list:
                    model_idx = list(range(len(gen_model)))
                    random.shuffle(model_idx)
                    chosen_model = gen_model[model_idx[0]]
                else:
                    chosen_model = gen_model
                gen_cap = copy.deepcopy(
                        data['captions']['dis'][filename[idx_batch[j]]][chosen_model])
                if len(gen_cap.shape) == 2:
                    gen_idx = range(gen_cap.shape[0])
                    random.shuffle(gen_idx)
                    x[j*2+1+len(idx_batch)*num_input, :] = gen_cap[gen_idx[0], :] # gen_idx[0]
                else:
                    x[j*2+1+len(idx_batch)*num_input, :] = gen_cap
            elif rand_ind == 1:  # MC samples
                """
                Not used in this reconstruction (comes from mc_samples: e.g. neuraltalk)
                """
                mc_cap = copy.deepcopy(
                        data['captions']['dis'][filename[idx_batch[j]]]['mc_samples']) # CHANGE MADE HERE 'mc_samples' not sure
                
                mc_idx = list(range(len(mc_cap)))
                random.shuffle(mc_idx) 
                mc_cap = mc_cap[mc_idx[0]]
                x[j*2+1+len(idx_batch)*num_input, :] = mc_cap
            elif rand_ind == 2:
                rand_ind_2 = np.random.choice(pathological_transf_idx)
                if rand_ind_2 == 0: # Random human caption
                    rand_j = np.random.randint(0,len(filename))
                    while rand_j == idx_batch[j]:
                        rand_j = np.random.randint(0,len(filename))
                    fake_cap = copy.deepcopy(data['captions']['dis'][filename[rand_j]]['human'])
                    if len(fake_cap.shape) > 1:
                        fake_idx = list(range(len(fake_cap)))
                        random.shuffle(fake_idx)
                        x[j*2+1+len(idx_batch)*num_input, :] = fake_cap[fake_idx[0]]
                    else:
                        x[j*2+1+len(idx_batch)*num_input, :] = fake_cap
                elif rand_ind_2 == 1: # random word replacement of human caption
                    human_cap = copy.deepcopy(
                            data['captions']['dis'][filename[idx_batch[j]]]['human'])
                    """
                    Authers has 5 people annotate each image, hence what they do here is to randomly select one of them
                    We don't have this 
                    """
                    if len(human_cap.shape) > 1:
                        human_idx = list(range(len(human_cap)))
                        random.shuffle(human_idx)
                        human_cap = human_cap[human_idx[0]] # This is for the original
                    if model._eos in list(human_cap):
                        end_position = list(human_cap).index(model._eos)
                    else:
                        end_position = len(human_cap) - 1
                    n_position = np.random.randint(min(2, end_position - 1), end_position)
                    rand_position = np.random.choice(end_position, size=(n_position,), replace=False)
                    rand_word = np.random.randint(config.vocab_size-4, size=(n_position,)) + 4
                    human_cap[rand_position] = rand_word
                    x[j*2+1+len(idx_batch)*num_input, :] = human_cap
 
                elif rand_ind_2 == 2: # random permutation of human captions
                    human_cap = copy.deepcopy(
                            data['captions']['dis'][filename[idx_batch[j]]]['human'])
                    if len(human_cap.shape) > 1:
                        human_idx = list(range(len(human_cap)))
                        random.shuffle(human_idx)
                        human_cap = human_cap[human_idx[0]]
                    if model._eos in list(human_cap):
                        end_position = list(human_cap).index(model._eos)
                    else:
                        end_position = len(human_cap) - 1
                    n_position = np.random.randint(min(2, end_position - 1), end_position)
                    rand_position = list(np.random.choice(end_position, size=(n_position,), replace=False))
                    rand_position_permutation = list(np.random.permutation(rand_position))
                    if rand_position_permutation == rand_position:
                        rand_position_permutation = list(np.random.permutation(rand_position))
                    human_cap[rand_position] = human_cap[rand_position_permutation]
                    x[j*2+1+len(idx_batch)*num_input, :] = human_cap
                else:
                    raise Exception("random number out of bound")
            else:
                raise Exception("random number out of bound")

            img[j*2+1+len(idx_batch)*num_input,:] = curr_img

        # feed_dict = {model.x: x, model.y_: y_, model.img_feat: img, model.lr : epoch_lr}
        effective_lr = config.learning_rate * config.learning_rate_decay ** epoch
        feed_dict = {model.x: x, model.y_: y_, model.img_feat: img, model.lr : effective_lr}

        vals = sess.run(fetches, feed_dict)
        loss = vals["loss"]
        accuracy = vals["accuracy"]

        if Config.resize_data:
            if verbose and (i % (epoch_size // 10) == 0 or i == epoch_size - 1):
                print("%d / %d loss: %.4f accuracy: %.3f speed: %.3f wps" %
                    (i + 1, epoch_size, loss, accuracy,
                    i * 1.0 * batch_size * num_steps / (time.time() - start_time)))
                loss_total.append(loss)
                accuracy_total.append(accuracy)
        else:
            if verbose and (i % (epoch_size // 10) == 10 or i == epoch_size - 1):
                print("%d / %d loss: %.4f accuracy: %.3f speed: %.3f wps" %
                    (i + 1, epoch_size, loss, accuracy,
                    i * 1.0 * batch_size * num_steps / (time.time() - start_time)))
                loss_total.append(loss)
                accuracy_total.append(accuracy)

    return loss_total, accuracy_total


def inference(sess, model, data, gen_model, dim_feat=2048, config=Config()):
    """Runs the model on the given data."""
    num_steps = config.num_steps
    num_input = config.num_input
    batch_size = config.batch_size
    if 'file_names' in data:
        filename = data['file_names']
    else:
        filename = data['image_ids']
    acc = []
    logits = []
    scores = []

    idx = list(range(len(filename)))
    epoch_size = len(idx) // batch_size
    if batch_size * epoch_size < len(idx):
        epoch_size += 1
        idx.extend(idx[:batch_size * epoch_size - len(idx)])

    for i in range(epoch_size):
        if i == epoch_size - 1:
            idx_batch = idx[batch_size*i:]
        else:
            idx_batch = idx[batch_size*i:batch_size*(i+1)]

        x = np.zeros((len(idx_batch)*num_input, num_steps), dtype=np.int32)
        y_ = np.zeros((len(idx_batch), 2), dtype=np.float32)
        y_[:, 1] = 1.0
        img = np.zeros((len(idx_batch)*num_input, dim_feat), dtype=np.float32)

        for j in range(len(idx_batch)):
            img_feat = copy.deepcopy(data['features']['dis'][filename[idx_batch[j]]])
            real_cap = copy.deepcopy(data['captions']['dis'][filename[idx_batch[j]]]['human'])
            real_idx = list(range(len(real_cap)))

            if len(real_cap.shape) > 1:
                random.shuffle(real_idx)
                x[j, :] = real_cap[real_idx[0]]
            else: 
                x[j, :] = real_cap
            img[j,:] = img_feat

            if gen_model == 'human':
                if len(real_cap.shape) > 1:
                    x[j+len(idx_batch), :] = real_cap[real_idx[1]]
                else: # ** THIS IS THE ONE APPLICABLE **
                    x[j+len(idx_batch), :] = real_cap
                y_[j, 0] = 1.
                y_[j, 1] = 0.
            elif gen_model == 'random_human':
                rand_j = random.randint(0,len(filename)-1)
                while rand_j == idx_batch[j]:
                    rand_j = random.randint(0,len(filename)-1)
                fake_cap = copy.deepcopy(data['captions']['dis'][filename[rand_j]]['human'])
                if len(fake_cap.shape) > 1:
                    fake_idx = list(range(len(fake_cap)))
                    random.shuffle(fake_idx)
                    x[j+len(idx_batch), :] = fake_cap[fake_idx[0]]
                else:
                    x[j+len(idx_batch), :] = fake_cap
            elif gen_model == 'random_word':
                x[j+len(idx_batch), :] = np.random.randint(
                        config.vocab_size-4, size=(num_steps,)) + 4
            else:
                x[j+len(idx_batch), :] = copy.deepcopy(
                        data['captions']['dis'][filename[idx_batch[j]]][gen_model])
            img[j+len(idx_batch),:] = img_feat
        acc_batch, logits_batch, scores_batch = sess.run([
            model._accuracy, model._logits, model._score],
            {model.x: x, model.y_: y_, model.img_feat:img})
        acc.append(acc_batch)
        logits.append(logits_batch)
        scores.append(scores_batch)

    print('%s Average Score: %.3f   Acc: %.3f' \
         % (gen_model, np.mean(np.array(scores)[:,:,0]), np.mean(np.array(acc))))

    return np.array(acc), np.array(logits), np.array(scores)

