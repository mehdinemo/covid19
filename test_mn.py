from tensorflow.keras.models import load_model
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import tensorflow as tf
from imutils import paths
import pandas as pd
import os
from tqdm import tqdm


def loaf_data(test_path: str):
    imagePaths = list(paths.list_images(test_path))

    index = []
    test_data = []
    # loop over the image paths
    for imagePath in tqdm(imagePaths):
        # extract the class label from the filename
        # ind = imagePath.split(os.path.sep)
        # ind = os.path.splitext(ind[1])[0]

        # ind = os.path.basename(imagePath)

        # load the image, swap color channels, and resize it to be a fixed
        # 224x224 pixels while ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        # update the data and labels lists, respectively
        test_data.append(image)
        index.append(imagePath)

    # convert the data and labels to NumPy arrays while scaling the pixel
    # intensities to the range [0, 255]
    test_data = np.array(test_data) / 255.0

    results = {'data': test_data, 'index': index}
    return results


def test_model():
    BS = 8

    imagePaths = list(paths.list_images('dataset'))
    labels = []
    data = []
    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]

        # load the image, swap color channels, and resize it to be a fixed
        # 224x224 pixels while ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)

    # convert the data and labels to NumPy arrays while scaling the pixel
    # intensities to the range [0, 255]
    data = np.array(data) / 255.0
    labels = np.array(labels)

    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    labels = to_categorical(labels)

    # load model from file
    model = load_model('covid19.model')

    predIdxs = model.predict(data, batch_size=BS)

    predIdxs = np.argmax(predIdxs, axis=1)

    print(classification_report(labels.argmax(axis=1), predIdxs,
                                target_names=lb.classes_))


def main():
    BS = 5

    test_path = 'test_data'

    print('load data')
    res = loaf_data(test_path)

    data = res['data']
    index = res['index']

    print(f'{len(data)} data loaded from {test_path}')

    model = load_model('covid19.model')

    predIdxs = model.predict(data, batch_size=BS)

    predIdxs_df = pd.DataFrame(predIdxs, index=index, columns=['covid', 'normal'])

    predIdxs_df.to_csv(f'test_results/predIdxs_df_{test_path}.csv')

    print('done')


if __name__ == '__main__':
    main()
