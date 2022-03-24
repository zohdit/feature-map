import os
import numpy as np
import tensorflow as tf
import vectorization_tools
from utils import move_distance, bitmap_count, orientation_calc
from feature import Feature
from sample import Sample
import json
from config import EXPECTED_LABEL
import matplotlib
matplotlib.use('Agg')
import itertools
from plot_utils import plot_heatmap
from pathlib import Path

BITMAP_THRESHOLD = 0.5


def _basic_feature_stats():
    return {
        'min' : np.PINF,
        'max' : np.NINF,
        'missing' : 0
    }

def extract_stats(x_test, y_test):
    features = ["moves", "bitmaps", "orientation"]
    # Iteratively walk in the dataset and process all the json files. For each of them compute the statistics
    data = {}
    # Overall Count
    data['total'] = 0
    # Features
    data['features'] = {f: _basic_feature_stats() for f in features}

    samples = []
    for seed in range(len(x_test)):
        if y_test[seed] == EXPECTED_LABEL:
            seed_image = x_test[seed]
            xml_desc = vectorization_tools.vectorize(seed_image)
            sample = Sample(xml_desc, y_test[seed], seed)
            performance = sample.evaluate()
            if performance < 0:
                misbehaviour = True
            else: 
                misbehaviour = False

            predicted_label = sample.predicted_label

            sample_dict = {
                "expected_label": str(y_test[seed]),
                "features": {
                    "moves":  move_distance(sample),
                    "orientation": orientation_calc(sample,0),
                    "bitmaps": bitmap_count(sample, BITMAP_THRESHOLD)
                },
                "id": sample.id,
                "misbehaviour": misbehaviour,
                "performance": str(performance),
                "predicted_label": predicted_label,
                "seed": seed 
            }
            seed_image = x_test[seed]
            xml_desc = vectorization_tools.vectorize(seed_image)
            sample =  Sample(xml_desc, EXPECTED_LABEL, seed)
            sample.from_dict(sample_dict)
            samples.append(sample)
            print(".", end='', flush=True)

            # Total count
            data['total'] += 1

            # Process only the features that are in the sample
            for k, v in data['features'].items():
                # TODO There must be a pythonic way of doing it
                if k not in sample_dict["features"].keys():
                    v['missing'] += 1

                    # if report_missing_features:
                    print("Sample %s miss feature %s", sample_dict["id"], k)

                    continue

                if sample_dict["features"][k] != "None":
                    v['min'] = min(v['min'], sample_dict["features"][k])
                    v['max'] = max(v['max'], sample_dict["features"][k])

    for feature_name, feature_extrema in data['features'].items():
        parsable_string_tokens = ["=".join(["name",feature_name])]
    for extremum_name, extremum_value in feature_extrema.items():
        parsable_string_tokens.append("=".join([extremum_name, str(extremum_value)]))
    print(",".join(parsable_string_tokens))

    filedest = "MNIST.meta"
    with open(filedest, 'w') as f:
        (json.dump(data, f, sort_keys=True, indent=4))

    return samples

def compute_featuremap_3d(features, samples):  
    
    # Generate the map axes
    map_features = []
    for f in features:
        print("Using feature %s" % f[0])
        map_features.append(Feature(f[0], f[1], f[2], f[3], f[4]))

    feature1 = map_features[0]
    feature2 = map_features[1]
    feature3 = map_features[2]

    # Reshape the data as ndimensional array. But account for the lower and upper bins.
    archive_data = np.full([feature1.num_cells, feature2.num_cells, feature3.num_cells], None, dtype=object)
    # counts the number of samples in each cell
    coverage_data = np.zeros(shape=(10,feature1.num_cells, feature2.num_cells, feature3.num_cells), dtype=int)

    misbehaviour_data = np.zeros(shape=(feature1.num_cells, feature2.num_cells, feature3.num_cells), dtype=int)

    for sample in samples:
        # Coordinates reason in terms of bins 1, 2, 3, while data is 0-indexed
        x_coord = feature1.get_coordinate_for(sample) - 1
        y_coord = feature2.get_coordinate_for(sample) - 1
        z_coord = feature3.get_coordinate_for(sample) - 1

        if archive_data[x_coord, y_coord, z_coord] is None:
            arx = [sample]
            archive_data[x_coord, y_coord, z_coord] = arx
        else:
            archive_data[x_coord, y_coord, z_coord].append(sample)
        # Increment the coverage 
        coverage_data[int(sample.expected_label),x_coord, y_coord, z_coord] += 1

        if sample.is_misbehavior():
            # Increment the misbehaviour 
            misbehaviour_data[x_coord, y_coord, z_coord] += 1

    return archive_data, coverage_data, misbehaviour_data

def compute_featuremap(features, samples):      
    # Generate the map axes
    map_features = []
    for f in features:
        print("Using feature %s" % f[0])
        map_features.append(Feature(f[0], f[1]))

    feature1 = map_features[0]
    feature2 = map_features[1]

    # Reshape the data as ndimensional array. But account for the lower and upper bins.
    archive_data = dict()
    # counts the number of samples in each cell
    coverage_data = dict()

    misbehaviour_data = dict()

    for sample in samples:
        x_coord = feature1.get_coordinate_for(sample) - 1
        y_coord = feature2.get_coordinate_for(sample) - 1

        # archive contains all inds in each cell
        if (x_coord, y_coord) not in archive_data:
            arx = [sample]
            archive_data[(x_coord, y_coord)] = arx
        else:
            archive_data[(x_coord, y_coord)].append(sample)

        # update the coverage 
        if (x_coord, y_coord) not in coverage_data:
            coverage_data[(x_coord, y_coord)] = sample
        else:
            if sample.performance < coverage_data[(x_coord, y_coord)].performance:
                coverage_data[(x_coord, y_coord)] = sample

        if sample.is_misbehavior():
            # Increment the misbehaviour 
            if (x_coord, y_coord) not in misbehaviour_data:
                misbehaviour_data[(x_coord, y_coord)] = 0
            else:
                misbehaviour_data[(x_coord, y_coord)] += 1

    return archive_data, coverage_data, misbehaviour_data, map_features


def extract_results(features, coverage_data):
        # self.log_dir_path is "logs/temp_..."
        log_dir_path = Path(f"logs")
        log_dir_path.mkdir(parents=True, exist_ok=True)


        archive_path = Path(f"logs/archive")
        archive_path.mkdir(parents=True, exist_ok=True)

        # filled values                                 
        filled = len(coverage_data)        

        original_seeds = set()
        mis_seeds = set()
        COUNT_MISS = 0
        for x in enumerate(coverage_data.items()): 
            # enumerate function returns a tuple in the form
            # (index, (key, value)) it is a nested tuple
            # for accessing the value we do indexing x[1][1]
            original_seeds.add(coverage_data[x[1][0]].seed)
            if float(x[1][1].performance) < 0:
                COUNT_MISS += 1
                mis_seeds.add(coverage_data[x[1][0]].seed)
            coverage_data[x[1][0]].export(archive_path)
    
        # feature_dict = dict()
        # for ft in features:
        #     feature_dict.update({f"{ft.name}_min": ft.min_value,
        #     f"{ft.name}_max": ft.ma_value})


        # Find the order of features in tuples
        b = tuple()
        for ft in features:
            i = ft.feature_name
            b = b + (i,)

        for feature1, feature2 in itertools.combinations(features, 2):         
            # # Create another folder insider the log one ...
            # log_dir_path = Path(f"{log_dir_name}/{feature1.name}_{feature2.name}")
            # log_dir_path.mkdir(parents=True, exist_ok=True)

            # Find the position of each feature in indexes
            x = list(b).index(feature1.feature_name)
            y = list(b).index(feature2.feature_name)

            # Define a new 2-D dict
            _solutions = {}
            _performances = {}
                
            for key, value in coverage_data.items():
                _key = (key[x], key[y])
                if _key in _performances:
                    if _performances[_key] > float(value.performance):
                        _performances[_key] = float(value.performance)
                        _solutions[_key] = coverage_data[key]
                else:
                    _performances[_key] = float(value.performance)
                    _solutions[_key] = coverage_data[key]

            # filled values                                 
            filled = len(_solutions)        

            original_seeds = set()
            mis_seeds = set()
            COUNT_MISS = 0
            for x in enumerate(_performances.items()): 
                # enumerate function returns a tuple in the form
                # (index, (key, value)) it is a nested tuple
                # for accessing the value we do indexing x[1][1]
                original_seeds.add(_solutions[x[1][0]].seed)
                if x[1][1] < 0:
                    COUNT_MISS += 1
                    mis_seeds.add(_solutions[x[1][0]].seed)
        

            str_performances = {}
            # convert keys to string   
            for key, value in _performances.items():
                str_performances[str(key)] = str(value)

            report = {
                'Covered seeds': len(original_seeds),
                'Filled cells': (filled),
                'Misclassified seeds': len(mis_seeds),
                'Misclassifications': COUNT_MISS,
                'Performances': str_performances
            }
            
            dst = f"{log_dir_path}/report_" + feature1.feature_name + "_" + feature2.feature_name + '.json'
            with open(dst, 'w') as f:
                (json.dump(report, f, sort_keys=False, indent=4))

            plot_heatmap(_performances,
                     feature1.feature_name,
                     feature2.feature_name,
                     savefig_path=str(log_dir_path)
                     )
  




if __name__ == "__main__":

    mnist = tf.keras.datasets.mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    samples =  extract_stats(x_test, y_test)

    features = []
    f3 = ["moves", "move_distance"]
    features.append(f3)
    f1 = ["bitmaps", "bitmap_count"]
    features.append(f1)


    archive_data, coverage_data, misbehaviour_data, map_features = compute_featuremap(features, samples)

    extract_results(map_features, coverage_data)

    features = []
    f3 = ["moves", "move_distance"]
    features.append(f3)
    f2 = ["orientation", "orientation_calc"]
    features.append(f2)


    archive_data, coverage_data, misbehaviour_data, map_features = compute_featuremap(features, samples)

    extract_results(map_features, coverage_data)

    features = []
    f2 = ["orientation", "orientation_calc"]
    features.append(f2)
    f1 = ["bitmaps", "bitmap_count"]
    features.append(f1)

    archive_data, coverage_data, misbehaviour_data, map_features = compute_featuremap(features, samples)

    extract_results(map_features, coverage_data)