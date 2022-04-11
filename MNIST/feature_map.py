import os
import sys

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf

import vectorization_tools
from config import EXPECTED_LABEL, META_FILE_DEST
from feature import Feature
from feature_simulator import FeatureSimulator
from sample import Sample
from utils import missing

matplotlib.use('Agg')
import itertools
import seaborn as sns
import matplotlib.pyplot as plt


def extract_samples_and_stats(data, labels):
    """
    Iteratively walk in the dataset and process all the json files.
    For each of them compute the statistics.
    """
    # Initialize the stats about the overall features
    stats = {feature_name: [] for feature_name in FeatureSimulator.get_simulators().keys()}

    data_samples = []
    filtered = list(filter(lambda t: t[1] == EXPECTED_LABEL, zip(data, labels)))
    for idx, item in enumerate(filtered):
        image, label = item
        xml_desc = vectorization_tools.vectorize(image)
        sample = Sample(xml_desc, label)
        data_samples.append(sample)
        # update the stats
        for feature_name, feature_value in sample.features.items():
            stats[feature_name].append(feature_value)

        # Show the progress
        sys.stdout.write('\r')
        progress = int((idx + 1) / len(filtered) * 100)
        progress_bar_len = 20
        progress_bar_filled = int(progress / 100 * progress_bar_len)
        sys.stdout.write(f'[{progress_bar_filled * "="}{(progress_bar_len - progress_bar_filled) * " "}]\t{progress}%')
        sys.stdout.flush()
    # New line after the progress
    print()

    stats = pd.DataFrame(stats)
    # compute the stats values for each feature
    stats = stats.agg(['min', 'max', missing, 'count'])
    stats.to_csv(META_FILE_DEST, index=False)
    print(stats.transpose())

    return data_samples, stats


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
    coverage_data = np.zeros(shape=(10, feature1.num_cells, feature2.num_cells, feature3.num_cells), dtype=int)

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
        coverage_data[int(sample.expected_label), x_coord, y_coord, z_coord] += 1

        if sample.is_misbehavior:
            # Increment the misbehaviour 
            misbehaviour_data[x_coord, y_coord, z_coord] += 1

    return archive_data, coverage_data, misbehaviour_data


def compute_featuremap(features, samples):
    print(f'Using the features {" + ".join([feature.feature_name for feature in features])}')

    feature1 = features[0]
    feature2 = features[1]

    # Reshape the data as ndimensional array. But account for the lower and upper bins.
    archive_data = np.full([feature1.num_cells, feature2.num_cells], None, dtype=object)
    # counts the number of samples in each cell
    coverage_data = np.zeros(shape=(feature1.num_cells, feature2.num_cells), dtype=int)

    misbehaviour_data = np.zeros(shape=(feature1.num_cells, feature2.num_cells), dtype=int)

    for sample in samples:
        x_coord = feature1.get_coordinate_for(sample) - 1
        y_coord = feature2.get_coordinate_for(sample) - 1

        if archive_data[x_coord, y_coord] is None:
            arx = [sample]
            archive_data[x_coord, y_coord] = arx
        else:
            archive_data[x_coord, y_coord].append(sample)
        # Increment the coverage 
        coverage_data[x_coord, y_coord] += 1

        if sample.is_misbehavior:
            # Increment the misbehaviour 
            misbehaviour_data[x_coord, y_coord] += 1

    return archive_data, coverage_data, misbehaviour_data


def visualize(features, samples):
    """
        Visualize the samples and the features on a map. The map cells contains the number of samples for each
        cell, so empty cells (0) are white, cells with few elements have a light color, while cells with more
        elements have darker color. This gives an intuition on the distribution of the misbheaviors and the
        collisions
    Returns:
    """

    figures = []
    # Create one visualization for each pair of self.axes selected in order
    for feature1, feature2 in itertools.combinations(features, 2):
        features_comb = [feature1, feature2]
        _, coverage_data, misbehaviour_data = compute_featuremap(features_comb, samples)

        # figure
        fig, ax = plt.subplots(figsize=(8, 8))

        cmap = sns.cubehelix_palette(dark=0.5, light=0.9, as_cmap=True)
        # Set the color for the under the limit to be white (so they are not visualized)
        cmap.set_under('1.0')

        # For some weird reason the data in the heatmap are shown with the first dimension on the y and the
        # second on the x. So we transpose
        coverage_data = np.transpose(coverage_data)

        sns.heatmap(coverage_data, vmin=1, vmax=20, square=True, cmap=cmap)

        # Plot misbehaviors - Iterate over all the elements of the array to get their coordinates:
        it = np.nditer(misbehaviour_data, flags=['multi_index'])
        for v in it:
            # Plot only misbehaviors
            if v > 0:
                alpha = 0.1 * v if v <= 10 else 1.0
                (x, y) = it.multi_index
                # Plot as scattered plot. the +0.5 ensures that the marker in centered in the cell
                plt.scatter(x + 0.5, y + 0.5, color="black", alpha=alpha, s=50)

        xtickslabel = [round(the_bin, 1) for the_bin in feature1.get_bins_labels()]
        ytickslabel = [round(the_bin, 1) for the_bin in feature2.get_bins_labels()]
        #
        ax.set_xticklabels(xtickslabel)
        plt.xticks(rotation=45)
        ax.set_yticklabels(ytickslabel)
        plt.yticks(rotation=0)

        title_tokens = ["Feature map: digit", str(EXPECTED_LABEL), "\n"]

        the_title = " ".join(title_tokens)

        fig.suptitle(the_title, fontsize=16)

        # Plot small values of y below.
        # We need this to have the y axis start from zero at the bottom
        ax.invert_yaxis()

        # axis labels
        plt.xlabel(feature1.feature_name)
        plt.ylabel(feature2.feature_name)

        # Add the store_to attribute to the figure object
        store_to = "-".join(["featuremap", str(EXPECTED_LABEL), feature1.feature_name, feature2.feature_name])
        setattr(fig, "store_to", store_to)

        figures.append(fig)

        file_format = 'pdf'

    for figure in figures:
        file_name_tokens = [figure.store_to]

        # Add File extension
        figure_file_name = "-".join(file_name_tokens) + "." + file_format

        figure_file = os.path.join("logs", figure_file_name)

        figure.savefig(figure_file, format=file_format)


if __name__ == "__main__":
    # Load the data
    mnist = tf.keras.datasets.mnist
    (_, _), (x_test, y_test) = mnist.load_data()
    # Extract the samples and the stats
    samples, stats = extract_samples_and_stats(x_test, y_test)
    # Get the list of features
    features = [
        Feature(feature_name, feature_stats['min'], feature_stats['max'])
        for feature_name, feature_stats in stats.to_dict().items()
    ]
    # Visualize the feature-maps
    visualize(features, samples)
