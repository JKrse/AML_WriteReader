# ============================================================================================================

from config import *
from pathlib import Path
import os 

# ============================================================================================================

arser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
        '--submission', type=str,
        help='Path to the json file that contains all the submissions.')
parser.add_argument(
        '--word-to-idx', type=str, default=f"{Config.local_path}/data/word_to_idx.npy",
        help='Path to the npy file that contains mapping from word to index.')
parser.add_argument(
        '--output-path', type=str, default=f"{Config.local_path}/data/",
        help='Path to the JSON file that contains how to split the dataset')
parser.add_argument(
        '--noicy', type=bool, default=False,
        help='If the data suppose to be noice (machine generated).')
parser.add_argument(
        '--num-steps', type=int, default=15,
        help='Length of all captions (default 15).')
parser.add_argument(
        '--name', type=str, default='mysubmission',
        help='Name of the method.')

args = parser.parse_args()
assert args.name != 'human' # Prevent naming conflits
assert args.name != 'mc_samples' # Prevent naming conflits

output_path = os.path.join(args.output_path, args.name)
# ============================================================================================================


name = Path("writereader")
output_path = Path(f"{Config.local_path}/data")

output_path =  Path(os.path.join(output_path, name))

if not os.path.exists(output_path):
    os.makedirs(output_path)

# If data has been augmentated 
if args.noicy == True: 
    name_placeholder = args.name
    args.name = "random_word_gen"


human_name = "human"
mc_name = "machine"

# ============================================================================================================

# Load splits, default to be Kaparthy's split
split = json.loads(open(args.split).read())
word2idx = np.load(args.word_to_idx, allow_pickle=True).item()

mysubmission = json.load(open(args.submission))

train_data = {}
val_data = {}
test_data = {}
train_filename = []
val_filename = []
test_filename = []

