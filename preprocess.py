import pickle

import numpy as np
import pandas as pd
from PIL import Image

label_df = pd.read_csv("./data/label_data.csv", index_col=0)


def process_data(lst: list) -> (list, list):
    length = len(lst)
    len_str = str(length)
    rtn_data = np.zeros((length, 100, 150, 3), dtype=np.float16)
    rtn_label = np.zeros((length, 4), dtype=np.uint8)
    for i, element in enumerate(lst):
        print("Processing " + str(i + 1) + " of " + len_str)
        folder = ['down', 'up', 'left', 'right'][label_df.loc[element, :].values.argmax()]
        rtn_data[i, :, :, :] = np.array(Image.open("./data/img_data/" + folder + "/" + element).convert('RGB'),
                                        dtype=np.int8) / 255
        rtn_label[i, :] = label_df.loc[element, :]
    return rtn_data, rtn_label


limit = int(label_df.shape[0] * 0.75)
train_data = label_df.index[:limit]
test_data = label_df.index[limit:]

print("Train Data")
X_train, y_train = process_data(train_data)
print("Test Data")
X_test, y_test = process_data(test_data)


def saving2disk(lst: np.ndarray, name: str):
    with open("./data/" + name, 'wb+') as fh:
        pickle.dump(lst, fh)
        print(name + " dumped.")


saving2disk(X_train, "X_train")
saving2disk(X_test, "X_test")
saving2disk(y_train, "y_train")
saving2disk(y_test, "y_test")
