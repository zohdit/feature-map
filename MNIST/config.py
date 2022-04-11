import json
from os.path import join

# Dataset
EXPECTED_LABEL = 5

IMG_SIZE = 28
NUM_CLASSES = 10
MODEL = 'models/tf2_model.h5'
META_FILE_DEST = 'logs/MNIST.meta'
FEATURES = ["Bitmaps", "Moves"]  # , "Orientation"]
NUM_CELLS = 25
BITMAP_THRESHOLD = 0.5
ORIENTATION_THRESHOLD = 0.


def to_json(folder):
    config = {
        'label': EXPECTED_LABEL,
        'image size': 28,
        'num classes': 10,
        'model': MODEL,
        'features': FEATURES,
        'num cells': 25
    }
    destination = join(folder, "config.json")
    with open(destination, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)
