import pickle as pkl
import numpy as np


def judgeGender(gender):
    if gender == '1':  # male
        return 1
    elif gender == '2':  # female 
        return 2
    else:  
        return 0


def loadLabelEncoder(address):
    with open(address, 'rb') as f:
        age = pkl.load(f)
        gender = pkl.load(f)
        province = pkl.load(f)
        label = pkl.load(f)
    return age, gender, province, label


def bias_score(true_label, predict_pro):
    ctr_gt = np.mean(true_label)
    ctr_predict = np.mean(predict_pro)
    ans = ctr_predict / ctr_gt - 1
    return ans

