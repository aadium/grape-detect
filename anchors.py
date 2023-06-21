import numpy as np
import os
from sklearn.cluster import KMeans

def load_dataset(train_dir, test_dir):
    train_dataset = load_folder(train_dir)
    test_dataset = load_folder(test_dir)
    return train_dataset, test_dataset

def load_folder(folder_dir):
    dataset = []

    for filename in os.listdir(folder_dir):
        if filename.endswith('.txt'):
            annotation_file = os.path.join(folder_dir, filename)
            image_file = os.path.join(folder_dir, filename.replace('.txt', '.jpg'))

            with open(annotation_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip().split()
                x_min, y_min, x_max, y_max = map(float, line[0:4])
                class_label = line[4]
                dataset.append([image_file, x_min, y_min, x_max, y_max, class_label])

    return dataset

from sklearn.cluster import KMeans
import numpy as np

def generate_anchors(num_anchors=9, output_path='anchors.txt'):
    train_dataset, test_dataset = load_dataset('train', 'test')
    dataset = train_dataset + test_dataset
    widths = []
    heights = []
    for data in dataset:
        x_min, y_min, x_max, y_max = data[1:5]
        width = x_max - x_min
        height = y_max - y_min
        widths.append(width)
        heights.append(height)

    # Perform K-means clustering to generate anchor boxes
    boxes = np.column_stack((widths, heights))
    kmeans = KMeans(n_clusters=num_anchors, random_state=42)
    kmeans.fit(boxes)

    # Get the coordinates of the anchor boxes
    anchors = kmeans.cluster_centers_

    # Save the anchor boxes to a file
    np.savetxt(output_path, anchors, delimiter=',')

    print(f"Generated {num_anchors} anchor boxes and saved them to {output_path}")

# Example usage
generate_anchors(num_anchors=9, output_path='anchors.txt')