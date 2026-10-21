#jiali zi
#20261021
import os
import torch
import datetime
from torch.utils.data import Dataset, DataLoader
from model.model_v3_LCFF_MLCA import mobilenet_v3_small
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import pandas as pd
import json
from model.Confu_maxtirx_fusion import ConfusionMatrix
from torch.nn import functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#JSON path
json_path = r"./class_indices_wildlife15.json"
# read JSON
with open(json_path, "r", encoding="utf-8") as f:
    raw_label_map = json.load(f)
LABEL_MAP = {v: int(k) for k, v in raw_label_map.items()}

# # Gibbon call detection
# LABEL_MAP = {
#     "absence": 0,
#     "presence": 1,
#     "partial presence": 2
# }

transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
def run_one_epoch(epoch_index, model, data_loader, optimizer, criterion, prin_req):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()
    avg_loss = 0
    true_total = 0
    total = 0
    for i, (mel_img, stft_img, labels) in enumerate(data_loader):
        mel_img, stft_img, labels = mel_img.to(device), stft_img.to(device), labels.to(device)
        outputs = model(mel_img, stft_img)
        loss = criterion(outputs, labels)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss += loss.item()
        pri_lab = outputs.argmax(-1)
        true_total += ((pri_lab == labels).sum().item())
        total += labels.size(0)

        if prin_req != 0 and (i + 1) % prin_req == 0:
            print("epoch %d  %s| step %d | avg loss %.3f | acc %.3f " % (
                epoch_index + 1, 'train' if is_train else 'eval', i + 1, loss.item(),
                true_total / total))
    return avg_loss / len(data_loader), true_total / total

def predict(test_loader, model, n_classes, save_path, cla_dict, file_dir, filename_time, save_mainName, epochs):
    # 1
    normalize = True
    model.load_state_dict(torch.load(save_path + f'{save_mainName}.pth', map_location=device))

    class_labels = [label for _, label in cla_dict.items()]
    confusion = ConfusionMatrix(num_classes=n_classes, labels=class_labels, normalize=normalize, batch_size=batch_size,
                                file_dir=file_dir, filename_time=filename_time, save_mainName=save_mainName,
                                epochs=epochs, model=model)
    model.to(device)
    model.eval()
    with torch.no_grad():
        total = 0
        true_total = 0
        for i, (mel_img, stft_img, labels) in enumerate(test_loader):
            mel_img, stft_img, labels = mel_img.to(device),stft_img.to(device), labels.to(device)
            outputs = model(mel_img, stft_img)
            pri_lab = outputs.argmax(-1)
            true_total += ((pri_lab == labels).sum().item())
            total += labels.size(0)

            confusion.update(pri_lab.to("cpu").numpy(), labels.to("cpu").numpy())

        print('test accuracy {:.3f} %'.format(100 * true_total / total))
    # confusion.plot()
    resultTable = confusion.summary()

    return true_total / total, resultTable

class AnimalSoundDataset(Dataset):
    def __init__(self, mel_root, stft_root, transform=None):

        self.transform = transform
        self.data_pairs = []  # (mel_path, stft_path, label)

        mel_base_dir = mel_root
        stft_base_dir = stft_root

        for class_name in os.listdir(mel_base_dir):
            class_mel_dir = os.path.join(mel_base_dir, class_name)
            class_stft_dir = os.path.join(stft_base_dir, class_name)
            label = LABEL_MAP[class_name]

            if not os.path.isdir(class_mel_dir):
                continue

            for filename in os.listdir(class_mel_dir):
                mel_path = os.path.join(class_mel_dir, filename)
                stft_path = os.path.join(class_stft_dir, filename)

                if os.path.exists(stft_path):
                    self.data_pairs.append((mel_path, stft_path, label))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        mel_path, stft_path, label = self.data_pairs[idx]

        mel_img = Image.open(mel_path).convert("RGB")
        stft_img = Image.open(stft_path).convert("RGB")

        if self.transform:
            mel_img = self.transform(mel_img)
            stft_img = self.transform(stft_img)

        return mel_img, stft_img, torch.tensor(label, dtype=torch.long)


# SimAM
class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.eps = 1e-8

    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (
                    4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda + self.eps)) + 0.5
        return x * self.act(y)

class SimAMFusion(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super(SimAMFusion, self).__init__()
        self.simam = SimAM()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, mel, stft):
        x1 = self.simam(mel)
        x2 = self.simam(stft)
        x = torch.cat((x1 * 0.8, x2 * 0.2), dim=1)  # (B, 6, H, W)
        x = self.conv1x1(x)
        return x
class MobileNetV3_Fusion(nn.Module):
    def __init__(self, num_classes=15): # num_classes = n_classes
        super(MobileNetV3_Fusion, self).__init__()
        self.fusion = SimAMFusion()
        self.mobilenet = mobilenet_v3_small()
        self.mobilenet.classifier[-1] = nn.Linear(self.mobilenet.classifier[-1].in_features, num_classes) 

    def forward(self, mel, stft):
        x = self.fusion(mel, stft)
        x = self.mobilenet(x)
        return x


if __name__ == '__main__':
    df = pd.DataFrame()
    # your dataset path
    train_mel_path = './dataset/logMel/train/'
    val_mel_path = './dataset/logMel/val/'
    test_mel_path = './dataset/logMel/test/'

    train_stft_path = './dataset/stft/train/'
    val_stft_path = './dataset/stft/val/'
    test_stft_path = './dataset/stft/test/'

    batch_size = 32
    epoch_size = 100
    LR = 0.001
    n_classes = 15
    print_freq = 20
    tmp_acc = 0
    best_acc = 0

    save_mainName = 'wildlife15DFC-MobV3'
    modelsave_path = './check_points/'
    file_dir = './result3/'

    train_dataset = AnimalSoundDataset(train_mel_path, train_stft_path, transform=transform)
    val_dataset = AnimalSoundDataset(val_mel_path, val_stft_path, transform=transform)
    test_dataset = AnimalSoundDataset(test_mel_path, test_stft_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=8, pin_memory=True)

    model = MobileNetV3_Fusion().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epoch_size):
        train_loss, train_aac = run_one_epoch(epoch_index=epoch, model=model, data_loader=train_loader,
                                              optimizer=optimizer, criterion=criterion, prin_req=print_freq)
        with torch.no_grad():
            val_loss, val_aac = run_one_epoch(epoch_index=epoch, model=model, data_loader=val_loader, optimizer=None,
                                              criterion=criterion, prin_req=0)
        # scheduler.step()
        if val_aac >= best_acc:
            torch.save(model.state_dict(), os.path.join(modelsave_path, f'{save_mainName}.pth'))
            best_acc = val_aac

        df = df._append(
            {'epoch_list': epoch, 'train_loss': train_loss,
             'train_acc': train_aac, 'val_loss': val_loss, 'val_acc': val_aac,
             'best_acc': best_acc}, ignore_index=True)

        print(
            f'epoch {epoch + 1} '
            f'| train avg loss : {train_loss}| train acc : {train_aac}'
            f'| test avg loss : {val_loss} | val acc : {val_aac}'
            f'| best acc : {best_acc}')

    now2 = datetime.datetime.now()
    filename_time = now2.strftime("%Y-%m-%d_%H-%M-%S")

    predictAcc, IndexTable = predict(test_loader, model, n_classes, modelsave_path, LABEL_MAP, file_dir, filename_time,
                                     save_mainName, epoch_size)
