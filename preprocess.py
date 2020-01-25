import pickle

import numpy as np
import pandas as pd
from PIL import Image

label_df = pd.read_csv("./label_data.csv", index_col=0)


def process_data(lst: list) -> (list, list):
    rtn_data = []
    rtn_label = []
    length = str(len(lst))
    for i, element in enumerate(lst):
        print("Processing " + str(i + 1) + " of " + length)
        image = np.array(Image.open("./data/all/" + element).convert('RGB'), dtype=np.int8) / 255
        image = (image - np.mean(image)) / np.std(image)
        rtn_data.append(image)
        rtn_label.append(label_df.loc[element, :])
    return rtn_data, rtn_label


limit = int(label_df.shape[0] * 0.75)
train_data = label_df.index[:limit]
test_data = label_df.index[limit:]

X_train, y_train = process_data(train_data)
X_test, y_test = process_data(test_data)


def saving2disk(lst: list, name: str) -> None:
    with open("./" + name, 'wb+') as fh:
        pickle.dump(lst, fh)


saving2disk(X_train, "X_train")
saving2disk(X_test, "X_test")
saving2disk(y_train, "y_train")
saving2disk(y_test, "y_test")
