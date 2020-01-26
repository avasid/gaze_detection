import os

import pandas as pd

dictt = {}
i = 0

for label in ['down', 'up', 'left', 'right']:
    img_lst = os.listdir("./data/img_data/" + label + "/")
    temp_label = [0] * 4
    temp_label[i] = 1
    for img in img_lst:
        print(label + " " + img)
        dictt[img] = temp_label
    i += 1

label_df = pd.DataFrame(data=dictt, index=['down', 'up', 'left', 'right']).transpose()
label_df = label_df.sample(frac=1)
label_df.to_csv("./data/label_data.csv")
