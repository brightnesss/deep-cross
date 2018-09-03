import torch
import torch.utils.data as data
import utlis
import linecache


class DataSetV2(data.Dataset):
    def __init__(self, file_dir, labelencoder_dir, is_train):
        super(DataSetV2, self).__init__()
        self.file_dir = file_dir
        data_len = 0
        with open(self.file_dir, 'r') as f:
            for line in f:
                data_len += 1
        self.data_len = data_len
        self.ageLabelEncoder, _, _, _ = utlis.loadLabelEncoder(labelencoder_dir)
        print("----going to cache all file----")
        l = linecache.getline(self.file_dir, 1)
        print("----finished cache------")

    def __len__(self):
        return self.data_len

    def __processLine__(self, line: str):
        """
        process each line str in data file
        do not process the category feature into one-hot feature

        the category features are all in the begin of feature list
        index 0: age    13
        index 1: gender     3
        index 2: similarity-image feature   1001
        index 3: color: warm or cold    2
        index 4: color: xian zhou   10
        index 5: color: ming an     10
        index 6: color: diversity   7
        index 7, 8, 9: color: top-3 main color  11

        from index 10, are the dense feature


        :param line: the str line from each row in data file
        :return: sparse_feature contains category, dense_feature contains float feature, label(either 0 or 1)
        """
        sparse_feature = []
        dense_feature = []
        line = line.split('\t')[:-1]
        line_after = line[-82:]

        label = int(line[6])

        # index 0, age
        sparse_feature.append(self.ageLabelEncoder.transform([line[18]])[0])
        # index 1, gender
        sparse_feature.append(utlis.judgeGender(line[19]))
        # index 2, similarity-image feature
        sparse_feature.append(1000 if line[21] == 'NULL' else int(line[21]))
        # index 3, warm or cold
        sparse_feature.append(1 if line_after[1] == '1' else 0)
        # index 4: color: xian zhou
        sparse_feature.append(int(line_after[3]))
        # index 5: color: ming an
        sparse_feature.append(int(line_after[4]))
        # index 6: color: diversity
        sparse_feature.append(int(line_after[5]))
        # index 7, 8, 9: color: top-3 main color
        for i in range(7, 10):
            sparse_feature.append(int(line_after[i]))

        # dense feature
        # sharpness
        dense_feature.append(float(line_after[0]))
        # warm / cold ratio
#        dense_feature.append(0.0 if float(line_after[2]) == 100000000 else float(line_after[2]))
        # diversity score
#        dense_feature.append(float(line_after[6]))
        # top-3 main color ratio
        for i in range(7, 10):
            dense_feature.append(float(line_after[i + 3]))
        # rest of feature from color, ocr and saliency
        for c in line_after[17:]:
            dense_feature.append(float(c))

        return sparse_feature, dense_feature, label

    def __getitem__(self, item):
        line = linecache.getline(self.file_dir, item + 1)
        sparse_feature, dense_feature, label = self.__processLine__(line)
        return torch.tensor(sparse_feature, dtype=torch.float), \
               torch.tensor(dense_feature, dtype=torch.float), \
               torch.tensor(label, dtype=torch.float)

