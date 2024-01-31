import argparse
import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch.optim as optim
from Network_for_colorproj import MLP

def randomdatas_generate():
    inputdt = np.random.rand(200,3)
    inputdt = torch.tensor(inputdt)
    outputdt = inputdt.clone
    outputdt = 3*(inputdt**2+1)
    return inputdt, outputdt

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0)
parser.add_argument("--endepoch", type=int, default=1000)
parser.add_argument("--dataset_name", type=str, default="Slcnn")
parser.add_argument("--batchsize", type=int, default=4)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--b1", type=float, default=0.99)
parser.add_argument("--output_model", type=str, default="D:\modelspace")
parser.add_argument("--randseed", type=int, default=14)
parser.add_argument("--device", type=str,default="cuda")
#parser.add_argument("--min", type=int, default=[-])
opt = parser.parse_args()

class CustomMSELoss(nn.Module):
    def __init__(self):
        super(CustomMSELoss, self).__init__()

    def forward(self, predicted, target):
        # 计算均方误差
        mse_loss = torch.mean((predicted - target)**2)
        return mse_loss

class Dataset_Generator(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        input_data = self.inputs[index].float()
        target_data = self.targets[index].float()
        return {'input':input_data, 'target': target_data}

def train_pipeline():
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(indt, oudt, test_size=0.2,random_state=opt.randseed)
    # inputs_train_mean = inputs_train.mean(axis=0)
    # inputs_train_std = inputs_train.std(axis=0)
    #
    # inputs_train = (inputs_train - inputs_train_mean) / inputs_train_std
    # inputs_test = (inputs_test - inputs_train_mean) / inputs_train_std

    train_set = Dataset_Generator(inputs_train, targets_train)
    trainloader = DataLoader(dataset=train_set, batch_size=opt.batchsize, shuffle=True)
    test_set = Dataset_Generator(inputs_test, targets_test)
    testloader = DataLoader(dataset=test_set, batch_size=opt.batchsize, shuffle=True)
    slight_model = MLP(3,[32,16,16,16,10],3).to(torch.device(opt.device))
    criterion = CustomMSELoss()
    optimizer = optim.Adam(slight_model.parameters(), lr=opt.lr)
    a=1
    train_epochs_loss = []
    val_epochs_loss = []
    best_val_loss = float('inf')

    for epoch in range(opt.endepoch):
        slight_model.train()
        train_epoch_loss = []
        acc, nums = 0, 0
        for i, batch in enumerate(tqdm(trainloader)):
            inputs = batch['input'].to(torch.device(opt.device))
            targets = batch['target'].to(torch.device(opt.device))
            outputs = slight_model(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            #loss += slight_model.l2_regularization()
            loss.backward(retain_graph=True)
            optimizer.step()
            train_epoch_loss.append(loss.item())
            nums += targets.size(0)
            #acc += torch.sum(torch.eq(predicted, targets)).item()

        train_epochs_loss.append(np.average(train_epoch_loss))
        avg_train_loss = sum(train_epoch_loss) / len(train_epoch_loss)
        accuracy = acc / nums
        #trainloader_iter = tqdm(trainloader, desc=f'Epoch {epoch + 1}/{opt.endepoch}, Loss: {avg_train_loss:.4f}, Accuracy: {accuracy:.4f}', position=0, leave=True)
        print(f'Epoch {epoch + 1}/{opt.endepoch}, Train Loss: {np.average(train_epoch_loss):.4f}, Accuracy: {accuracy:.4f}')

        with torch.no_grad():
            slight_model.eval()
            val_epoch_loss = []
            acc, nums = 0, 0

            for i, batch in enumerate(tqdm(testloader)):
                inputs = batch['input'].to(torch.device(opt.device))
                targets = batch['target'].to(torch.device(opt.device))
                outputs = slight_model(inputs)
                optimizer.zero_grad()
                loss = criterion(outputs, targets)
                loss.requires_grad_(True)
                # loss += slight_model.l2_regularization()
                loss.backward(retain_graph=True)
                optimizer.step()
                val_epoch_loss.append(loss.item())

            val_epochs_loss.append(np.average(val_epoch_loss))
            print(f'Test Loss: {np.average(val_epoch_loss):.4f}')

        if np.average(val_epoch_loss) < best_val_loss:
            # best_val_loss = np.average(val_epoch_loss)
            # best_model_state = slight_model.state_dict()
            # best_optimizer_state = optimizer.state_dict()
            torch.save(slight_model.state_dict(), 'best_model.pt')
            best_val_loss = np.average(val_epoch_loss)

    print(f'best loss is {best_val_loss:.4f}')
    # torch.save({
    #     'model_state_dict': best_model_state,
    #     'optimizer_state_dict': best_optimizer_state,
    #     'best_val_loss': best_val_loss,
    # }, 'best_model.pt')
    # print(f'best loss:{best_val_loss:.4f}')

def pred(val):
    smodel = MLP(3, [32,16,16,16,10], 3)
    # optimizer = optim.SGD(smodel.parameters(), lr=0.01)

    checkpoint = smodel.load_state_dict(torch.load('best_model.pt'), False)
    # smodel.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    smodel.eval()
    val = torch.tensor(val).reshape(1, -1).float()
    res = smodel(val)
    return res

if __name__ == '__main__':
    # [idt, odt] = randomdatas_generate()
    # indt = torch.tensor(idt)
    # oudt = torch.tensor(odt)

    #训练模型------------------------------------------------------------
    # indt = torch.tensor(np.loadtxt('rgbs.txt'))
    # oudt = torch.tensor(np.loadtxt('xyzs.txt'))
    # train_pipeline()
    # #-------------------------------------------------------------------

    #测试性能-------------------------------------------------------------
    rgbtst = np.loadtxt('rgbtst.txt')
    xyztst = torch.tensor(rgbtst.copy())
    for i in range(rgbtst.shape[0]):
        xyztst[i] = pred(torch.tensor(rgbtst[i]))
    xyztst = xyztst.detach().numpy()
    np.savetxt('xyzpre.txt',xyztst)
   # np.savetxt(r'C:\Users\Administrator\PycharmProjects\LUT\venv\xyztst.txt', xyztst)
    #--------------------------------------------------------------------
