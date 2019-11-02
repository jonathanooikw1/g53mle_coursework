# This script only has to be run once to generate the data .txt files
import h5py
import numpy as np
import os
import csv
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

features_directory = "../BP4D/2DFeatures"
labels_directory = "../BP4D/Labels/OCC"


# This function takes in the list of x or y-coordinates for
# one image, and makes the range from 0 to 1
def normalize_coordinate(coordinate):
    # Using MinMaxScaler API
    coordinate = np.reshape(coordinate, (-1, 1))
    scalar = MinMaxScaler(feature_range=(0, 1))
    scalar = scalar.fit(coordinate)
    normalized = scalar.transform(coordinate)
    normalized = np.reshape(normalized, -1)
    return normalized


# This function takes in the list of coordinates
# Splits it into x and y coordinates for normalization
# And then recombines it to return the normalized coordinates
def normalize_coordinates(coordinates):
    x_coordinates = []
    y_coordinates = []
    # Splitting coordinates into x and y
    for coordinate in coordinates:
        x_coordinates.append(coordinate[0])
        y_coordinates.append(coordinate[1])
    # Normalizing x and y separately
    x_coordinates = normalize_coordinate(x_coordinates)
    y_coordinates = normalize_coordinate(y_coordinates)
    normalized_coordinates = []
    # Recombining the x and y coordinates
    for i in range(0, len(x_coordinates)):
        normalized_coordinates.append([x_coordinates[i], y_coordinates[i]])
    return normalized_coordinates


# This function takes in a single .mat file and the corresponding .csv file
# and returns the feature list and the label list
def feature_and_label_from_file(file_name):
    labels_file_name = os.path.join(labels_directory, file_name.replace(".mat", ".csv"))
    feature_file_name = os.path.join(features_directory, file_name)

    fh = h5py.File(feature_file_name, 'r')  # Initializing h5py file handler
    lms_obj = fh.get('fit/pred')  # Extracting the list of landmarks array objects from pred field
    feature_list = []
    label_list = []
    with open(labels_file_name, newline='') as csv_file:
        reader = csv.reader(csv_file)
        first = True
        for row in reader:
            # Skip first header row of csv file
            if first:
                first = False
                continue
            # Skips data points that are not (49,2)
            if len(fh[lms_obj[int(row[0])][0]][()].transpose()) == 49:
                # row[0] is the image number, not the label
                # only append row 1 to row 6
                label_list.append(list(map(int, row[1:6])))
                # Obtains the features corresponding to the image number and normalizes it
                normalized_coordinates = normalize_coordinates(fh[lms_obj[int(row[0])][0]][()].transpose())
                # This flattens the coordinates from 49,2 to 98 as the neural network
                # only takes inputs of one dimension
                normalized_coordinates = np.reshape(normalized_coordinates, -1)
                feature_list.append(normalized_coordinates)

    return feature_list, label_list


# This function takes in the .mat directory
# Returns the label and feature list
def feature_and_label_from_directory(directory=features_directory):
    label_list = []
    feature_list = []
    for fileName in os.listdir(directory):
        # To ignore clipboard.txt
        if fileName.find('.mat') != -1:
            print(fileName)
            temp_feature, temp_label = feature_and_label_from_file(fileName)
            label_list.extend(temp_label)
            feature_list.extend(temp_feature)
    return feature_list, label_list


def generate_data():
    feature_list, label_list = feature_and_label_from_directory()
    # Converts list to array, easier to work with
    feature_array = np.asarray(feature_list)
    label_array = np.asarray(label_list)
    # Shuffles the data
    feature_array, label_array = shuffle(feature_array, label_array)
    # Splits the data into 80 training 20 test
    X_train, X_test, y_train, y_test = train_test_split(feature_array, label_array, test_size=0.20, random_state=42)
    print("Feature train shape: " + str(X_train.shape))
    print("Label train shape: " + str(y_train.shape))
    print("Feature test shape: " + str(X_test.shape))
    print("Label test shape: " + str(y_test.shape))
    # Saves data as txt file
    np.savetxt('X_train.txt', X_train, fmt='%s')
    np.savetxt('X_test.txt', X_test, fmt='%s')
    np.savetxt('y_train.txt', y_train, fmt='%d')
    np.savetxt('y_test.txt', y_test, fmt='%d')


generate_data()
