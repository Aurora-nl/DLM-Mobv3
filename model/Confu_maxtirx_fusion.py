import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import itertools
from thop import profile
import torch
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# FLOPs
def measure_flops(model, input_tensor):
    mel, cwt = input_tensor
    flops, params = profile(model, inputs=(mel, cwt))
    return flops, params

# Latency
def measure_latency(model, input_tensor, num_runs=100):
    mel, cwt = input_tensor
    for _ in range(10):
        _ = model(mel,cwt)
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(mel,cwt)
    end_time = time.time()

    latency = (end_time - start_time) / num_runs  # 平均时延
    return latency

# Throughput
def measure_throughput(model, input_batch, num_runs=100):
    mel_batch, cwt_batch = input_batch
    for _ in range(10):
        _ = model(mel_batch,cwt_batch)

    start_time = time.time()
    for _ in range(num_runs):
        _ = model(mel_batch, cwt_batch)
    end_time = time.time()

    total_samples = mel_batch.size(0) * num_runs
    throughput = total_samples / (end_time - start_time)  # 每秒处理的样本数
    return throughput

class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list, normalize: bool, batch_size: int,file_dir,filename_time,save_mainName,epochs,model):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels
        self.normalize = normalize
        self.batch_size = batch_size
        self.file_dir = file_dir
        self.filename_time = filename_time
        self.save_mainName = save_mainName
        self.epochs = epochs
        self.model = model

    def indexSome(self, preds, labels):
        return
    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        table = PrettyTable()
        PrecisionSum = 0
        RecallSum = 0
        F1scoreSum = 0
        SpecificitySum = 0
        table.field_names = ["", "Precision", "Recall", "Specificity","f1_score","flops","params","latency","throughput"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.

            f1_score = 2 * (Precision * Recall) / (Precision + Recall) if Precision + Recall > 0 else 0
            PrecisionSum = PrecisionSum + Precision
            RecallSum = RecallSum + Recall
            SpecificitySum = SpecificitySum + Specificity
            F1scoreSum = F1scoreSum + f1_score

            table.add_row([self.labels[i], Precision, Recall, Specificity,f1_score,0,0,0,0])

        PrecisionMean = PrecisionSum / self.num_classes
        RecallMean =  RecallSum / self.num_classes
        F1scoreMean = F1scoreSum / self.num_classes
        SpecificityMean = SpecificitySum / self.num_classes

        mel_input_tensor = torch.randn(1, 3, 224, 224).to(device)
        stft_input_tensor = torch.randn(1, 3, 224, 224).to(device)

        input_batch_mel = torch.randn(32, 3, 224, 224).to(device)
        input_batch_cwt = torch.randn(32, 3, 224, 224).to(device)

        flops, params = measure_flops(self.model, (mel_input_tensor,stft_input_tensor))
        # Latency
        latency = measure_latency(self.model, (mel_input_tensor,stft_input_tensor))
        # Throughput
        throughput = measure_throughput(self.model, (input_batch_mel, input_batch_cwt))
        table.add_row(["MeanValue", PrecisionMean, RecallMean, SpecificityMean, F1scoreMean,flops/ 1e9 ,params/ 1e6,latency* 1000,throughput])

        indexFilaname = self.file_dir + self.save_mainName + '-epoch' + str(self.epochs) + "-indexValue.txt"
        with open(indexFilaname, 'w') as f:
            f.write(table.get_string())

        print(table)
        print("The Final Result MeanValue: PrecisionMean = {}, RecallMean = {}, F1scoreMean = {},flops = {},params = {},latency = {},throughput = {} .".format(PrecisionMean,
                                                                           RecallMean, F1scoreMean,flops,params,latency,throughput))

    def plot(self):
        # print(matrix)
        self.plot_confusion_matrix()

    def plot_confusion_matrix(self):
        matrix = self.matrix
        classes = self.labels
        normalize = self.normalize

        fig = plt.figure(1)
        title = 'Confusion matrix'
        cmap = plt.cm.Blues

        print('normalize: ', normalize)
        if normalize:
            matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
            print("Display percentage: ")
            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
            # print(matrix)
        else:
            print('Display specific numbers: ')
            # print(matrix)
        plt.imshow(matrix, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, ha='right')
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else '.0f'
        thresh = matrix.max() / 2.
        for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
            plt.text(j, i, format(matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if matrix[i, j] > thresh else "black")
        plt.tight_layout()
        plt.gcf().subplots_adjust(bottom=0.3)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.savefig(self.file_dir + self.save_mainName + '-epoch' + str(self.epochs) + "-confusion_matrix.png")
        plt.show()
        plt.close(1)
