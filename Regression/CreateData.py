import h5py
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

features_directory = "2DFeatures"
labels_directory = "Labels/OCC"


def normalize_coordinate(coordinate):
    coordinate = np.reshape(coordinate, (-1, 1))
    scalar = MinMaxScaler(feature_range=(0, 1))
    scalar = scalar.fit(coordinate)
    normalized = scalar.transform(coordinate)
    normalized = np.reshape(normalized, -1)
    return normalized


def normalize_coordinates(coordinates):
    x_coordinates = []  # x coordinates array
    y_coordinates = []  # y coordinates array

    for coordinate in coordinates:
        x_coordinates.append(coordinate[0])
        y_coordinates.append(coordinate[1])

    x_coordinates = normalize_coordinate(x_coordinates)
    y_coordinates = normalize_coordinate(y_coordinates)
    normalized_coordinates = []
    for i in range(0, len(x_coordinates)):
        normalized_coordinates.append([x_coordinates[i], y_coordinates[i]])
    return normalized_coordinates


def grab_features_and_label():
    feature_list = []
    label_list = []
    for fileName in os.listdir(features_directory):
        if fileName.find('.mat') != -1:
            print(fileName)
            temp_feature_list = []
            temp_label_list = []
            current_file = features_directory + "/" + fileName
            fh = h5py.File(current_file, 'r')
            lms_feature = fh.get('fit/pred')
            lms_angle = fh.get('fit/pose')
            '''all_feature_array = np.zeros((len(lms_feature), 98))
            all_angle_array = np.zeros((len(lms_angle), 3))'''
            for i in range(0, len(lms_feature)):
                if fh[lms_feature[i][0]].size != 2:
                    if type(fh[lms_angle[i][0]]) is not h5py._hl.dataset.Dataset:
                        temp_list1 = normalize_coordinates(fh[lms_feature[i][0]].value.transpose())
                        #all_feature_array[i] = np.reshape(temp_list, -1)
                        temp_list1 = np.reshape(temp_list1, -1)
                        rot_array = fh[lms_angle[i][0]].get('angle')
                        temp_list2 = rot_array.value.transpose()
                        temp_list2 = np.reshape(temp_list2, -1)
                        temp_feature_list.append(temp_list1)
                        temp_label_list.append(temp_list2)
                        #all_angle_array[i] = rot_array.value.transpose()
            feature_list.extend(temp_feature_list)
            label_list.extend(temp_label_list)
    return feature_list, label_list


'''def grab_labels():
    label_list = []
    for fileName in os.listdir(features_directory):
        if fileName.find('.mat') != -1:
            print(fileName)
            temp_label_list = []
            current_file = features_directory + "/" + fileName
            fh = h5py.File(current_file, 'r')
            lms_angle = fh.get('fit/pose')
            all_angle_array = np.zeros((len(lms_angle), 3))
            for i in range(0, len(lms_angle)):
                if type(fh[lms_angle[i][0]]) is not h5py._hl.dataset.Dataset:
                    #  print(fileName + ' array type is ', fh[lms_angle[i][0]])
                    rot_array = fh[lms_angle[i][0]].get('angle')
                    all_angle_array[i] = rot_array.value.transpose()
            temp_label_list.append(all_angle_array)
            label_list.extend(temp_label_list)

    return label_list'''


def generateData():
    all_features, all_angles = grab_features_and_label()
    all_angles = np.asarray(all_angles)
    feature_array, label_array = shuffle(all_features, all_angles)
    X_train, X_test, y_train, y_test = train_test_split(feature_array, label_array, test_size=0.20, random_state=42)
    np.savetxt('X_train.txt', X_train, fmt='%s')
    np.savetxt('X_test.txt', X_test, fmt='%s')
    np.savetxt('y_train.txt', y_train, fmt='%s')
    np.savetxt('y_test.txt', y_test, fmt='%s')


generateData()
