
from pathlib import Path 
import os


class Config(object):
    """ Model Configuration
    [use_img_feat] options control how we use image features
        None                  not using image features (default)
        "concat_bf_lstm"      concatenate image feature before LSTM
        "concat_af_lstm"      concatenate image feature after LSTM
        "only_img"            image feature only
    [combine_typ]  how do we combine context feature with candidate feature
        "bilinpool"  using binlinear pooling (default)
        "concat"      concatenate features directly
    [cls_hidden]   number of hidden layers for classifer (all with size 256)
    """
    learning_rate = 0.0008
    learning_rate_decay = 0.9
    max_epoch = 10 #
    grad_clip = 1.0
    num_layers = 1
    num_steps = 15
    hidden_size = 512
    dropout_prob = 0.6
    batch_size = 32
    vocab_size = 10004
    embedding_size = 300
    num_input = 2
    use_lstm  = True

    # Utilize random search: 
    random_search = False

    # Resize training data:
    resize_data = False
    resize_samples = 800

    # Path for terminal / debugging locally
    local_path = Path("./local_files/")
    local_path_temp = Path("./../local_files/")


    # How to use Image Feature :
    #   None | 'concat_bf_lstm' | 'concat_af_lstm' | 'only_img'
    use_img_feat= 'only_img'

    # How to combine context feature:
    #   'bilinpool' | 'concat'
    combine_typ = 'concat'

    # 0 for basic linear classifier
    cls_hidden = 0
    use_residual         = False # Whether use residual connection in LSTM
    use_random_human     = True  # Whether using random human captions transformations
    use_random_word      = True  # Whether using random word replacement
    use_word_permutation = True  # Whehter using random word permutations
    use_mc_samples       = True  # Whether using Monte Carlo Sampled Captions



def environment_setup():
    """
    Generate the folder structure needed for project.
    """
    if not os.path.exists(Config.local_path):
        os.makedirs(Config.local_path)
    if not os.path.exists(Config.local_path / "data"):
        os.makedirs(Config.local_path / "data")
    if not os.path.exists(Config.local_path / "figures"):
        os.makedirs(Config.local_path / "figures")
    if not os.path.exists(Config.local_path / "images"):
        os.makedirs(Config.local_path / "images")

# if not os.path.exists(Config.local_path):
#     environment_setup()



def config_model_coco(config, model_architecture):
    config.num_layers = 1 # using 1 LSTM layer
    # Linear models
    if model_architecture == 'concat_no_img_1_512_0':
        config.use_img_feat = None
        config.combine_typ = 'concat'
        config.hidden_size = 512
        config.cls_hidden = 0
    elif model_architecture == 'concat_img_1_512_0':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'concat'
        config.hidden_size = 512
        config.cls_hidden = 0
    elif model_architecture == 'concat_only_img_1_512_0':
        config.use_img_feat = 'only_img'
        config.combine_typ = 'concat'
        config.hidden_size = 512
        config.cls_hidden = 0
    elif model_architecture == 'concat_img_1_512_0_noda':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'concat'
        config.hidden_size = 512
        config.cls_hidden = 0
        config = set_no_da(config)
    # Non-linear models with Compact Bilinear Pooling
    elif model_architecture == 'bilinear_img_1_512_0':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'bilinpool'
        config.hidden_size = 512
        config.cls_hidden = 0
    elif model_architecture == 'bilinear_no_img_1_512_0':
        config.use_img_feat = None
        config.combine_typ = 'bilinpool'
        config.hidden_size = 512
        config.cls_hidden = 0
    elif model_architecture == 'bilinear_only_img_1_512_0':
        config.use_img_feat = 'only_img'
        config.combine_typ = 'bilinpool'
        config.hidden_size = 512
        config.cls_hidden = 0
    elif model_architecture == 'bilinear_img_1_512_0_noda':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'bilinpool'
        config.hidden_size = 512
        config.cls_hidden = 0
        config = set_no_da(config)
    # Non-linear models with MLP
    elif model_architecture == 'mlp_1_img_1_512_0':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'concat'
        config.hidden_size = 512
        config.cls_hidden = 1
    elif model_architecture == 'mlp_1_no_img_1_512_0':
        config.use_img_feat = None
        config.combine_typ = 'concat'
        config.hidden_size = 512
        config.cls_hidden = 1
    elif model_architecture == 'mlp_1_only_img_1_512_0':
        config.use_img_feat = 'only_img'
        config.combine_typ = 'concat'
        config.hidden_size = 512
        config.cls_hidden = 1
    elif model_architecture == 'mlp_1_img_1_512_0_noda':
        config.use_img_feat = 'concat_af_lstm'
        config.combine_typ = 'concat'
        config.hidden_size = 512
        config.cls_hidden = 1
        config = set_no_da(config)
    else:
        raise Exception("Invalid architecture name:%s"%model_architecture)
    return config
