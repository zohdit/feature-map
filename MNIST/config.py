from os.path import join
import json

# Dataset
EXPECTED_LABEL   = 5

IMG_SIZE         = 28
NUM_CLASSES      = 10
MODEL            = 'models/tf2_model.h5'
BITMAP_THRESHOLD = 0.5
FEATURES         = ["Bitmaps", "Moves"] #, "Orientation"]
NUM_CELLS        = 25


def to_json(folder):
    config = {
        'label': str(EXPECTED_LABEL),
        'image size': 28,
        'num classes' : 10,
        'model': str(MODEL),
        'features': str(FEATURES),
        'num cells': 25
    }
    filedest = join(folder, "config.json")
    with open(filedest, 'w') as f:
        (json.dump(config, f, sort_keys=True, indent=4))
