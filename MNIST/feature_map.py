import tensorflow as tf

from feature import Feature
from utils.feature_map.preprocess import extract_samples_and_stats
from utils.feature_map.visualize import visualize_map

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
    visualize_map(features, samples)
