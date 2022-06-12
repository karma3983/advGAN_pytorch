import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models
from models import MNIST_target_net

use_cuda=True
image_nc=1
batch_size = 128

gen_input_nc = image_nc

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# load the pretrained model 学習済みモデルをロードする
pretrained_model = "./MNIST_target_model.pth"
target_model = MNIST_target_net().to(device)
target_model.load_state_dict(torch.load(pretrained_model))
target_model.eval()

# load the generator of adversarial examples 敵対的サンプルのジェネレータをロードする
pretrained_generator_path = './models/netG_epoch_60.pth'
pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

# test adversarial examples in MNIST training datase　MNISTトレーニングデータセットに含まれる敵対的サンプルを学習する
#基本的にtrain_target_model.pyと同じ
mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
train_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
num_correct = 0
#enumerate()のカッコ内にリスト(train_dataloader)等を指定、iとdataに番号(0,1,2,...)と要素を格納、開始値は0に指定
for i, data in enumerate(train_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    #pretrained_G = models.Generator(gen_input_nc, image_nc).to(device) なお、gen_input_nc=1、image_nc=1
    perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -0.3, 0.3) #上限0.3、下限-0.3にTensorをクランプ(元々範囲内なら効果なし)
    adv_img = perturbation + test_img
    adv_img = torch.clamp(adv_img, 0, 1) #上限1、下限0にTensorをクランプ(元々範囲内なら効果なし)
    #いつもならtest_imgを使用する、今回は敵対的サンプルと一致しているかなのでadv_img
    pred_lab = torch.argmax(target_model(adv_img),1) #target_model = MNIST_target_net().to(device)　なお、dim(縮小する次元)=1
    num_correct += torch.sum(pred_lab==test_label,0) #予測値＝ラベル、pred_lab==test_labelがdimを保持しているか確認

    print("番号：",i)
    print("画像",test_img)
    print("ラベル：",test_label)
    print("ノイズ：",perturbation)
    print("ノイズ＋画像：",adv_img)
    print("予測値：",pred_lab)
    print("正解値：",num_correct)
    print("-------------------------------------------")

print('MNIST training dataset:')
print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/len(mnist_dataset)))  # num_correct.item() ÷ 60000

# test adversarial examples in MNIST testing dataset　MNISTデータセットでテストする
#基本的にtrain_target_model.pyと同じ
mnist_dataset_test = torchvision.datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True) #train = Falseバージョン
test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)
num_correct = 0
for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    #pretrained_G = models.Generator(gen_input_nc, image_nc).to(device) なお、gen_input_nc=1、image_nc=1
    perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -0.3, 0.3) #上限0.3、下限-0.3にTensorをクランプ(元々範囲内なら効果なし)
    adv_img = perturbation + test_img
    adv_img = torch.clamp(adv_img, 0, 1) #上限1、下限0にTensorをクランプ(元々範囲内なら効果なし)
    #いつもならtest_imgを使用する、今回は敵対的サンプルと一致しているかなのでadv_img
    pred_lab = torch.argmax(target_model(adv_img),1) #target_model = MNIST_target_net().to(device)　なお、dim(縮小する次元)=1
    num_correct += torch.sum(pred_lab==test_label,0) #予測値＝ラベル、pred_lab==test_labelがdimを保持しているか確認

print('MNIST test dataset:')
print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in testing set: %f\n'%(num_correct.item()/len(mnist_dataset_test)))  # num_correct.item() ÷ 10000

