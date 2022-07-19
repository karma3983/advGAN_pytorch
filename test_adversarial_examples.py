import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models
from models import MNIST_target_net
import matplotlib.pyplot as plt

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
pretrained_generator_path = './models/netG_epoch_60.pth' #本来はこっち----------------------------------------------------------------------------------------
#pretrained_generator_path = './models/netG_epoch_3.pth'
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

    #print("番号：",i)
    #print("画像",test_img)
    #print("ラベル：",test_label)
    #print("ノイズ：",perturbation)
    #print("ノイズ＋画像：",adv_img)
    #print("予測値：",pred_lab)
    #print("正解値：",num_correct)
    #print("-------------------------------------------")

print('MNIST training dataset:')
print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/len(mnist_dataset)))  # num_correct.item() ÷ 60000

for i in train_dataloader:
  print("real")
  plt.imshow(i[0][0].reshape(28,28))
  plt.show()
  #real_inputs = i[0][0]
  #noise = (torch.rand(real_inputs.shape[0], 128)-0.5)/0.5
  #noise = noise.to(device)
  #fake_inputs = G(noise)
  #print("fake")
  #plt.imshow(fake_inputs[0][0].cpu().detach().numpy().reshape(28,28))
  #plt.show()
    
  plt.plot([1,2,3,4])
  plt.show()
  break

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

#番号： 468
#画像 tensor([[[[0., 0., 0.,  ..., 0., 0., 0.],
#        省略
#          ...,
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.]]]], device='cuda:0')
#ラベル： tensor([3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
#        6, 0, 3, 4, 1, 4, 0, 7, 8, 7, 7, 9, 0, 4, 9, 4, 0, 5, 8, 5, 9, 8, 8, 4,
#        0, 7, 1, 3, 5, 3, 1, 6, 5, 3, 8, 7, 3, 1, 6, 8, 5, 9, 2, 2, 0, 9, 2, 4,
#        6, 7, 3, 1, 3, 6, 6, 2, 1, 2, 6, 0, 7, 8, 9, 2, 9, 5, 1, 8, 3, 5, 6, 8],
#       device='cuda:0')
#ノイズ： tensor([[[[-1.0992e-02, -4.5701e-03, -1.8996e-02,  ...,  5.3821e-02,
#        省略
#          ...,
#          [-3.5965e-03, -4.3009e-03,  5.5456e-03,  ...,  1.4112e-03,
#           -1.0565e-02,  2.1239e-04]]]], device='cuda:0',
#       grad_fn=<ClampBackward1>)
#ノイズ＋画像： tensor([[[[0.0000e+00, 0.0000e+00, 0.0000e+00,  ..., 5.3821e-02,
#        省略
#          ...,
#          [0.0000e+00, 0.0000e+00, 5.5456e-03,  ..., 1.4112e-03,
#           0.0000e+00, 2.1239e-04]]]], device='cuda:0',
#       grad_fn=<ClampBackward1>)
#予測値： tensor([8, 8, 2, 7, 3, 3, 3, 2, 3, 8, 8, 3, 3, 3, 2, 3, 8, 8, 8, 2, 3, 5, 3, 3,
#        3, 2, 8, 8, 3, 8, 8, 3, 3, 3, 3, 3, 2, 3, 3, 8, 2, 2, 3, 2, 3, 3, 3, 8,
#        2, 9, 3, 0, 2, 7, 3, 2, 2, 8, 3, 3, 8, 3, 2, 3, 2, 3, 8, 8, 2, 3, 8, 8,
#        2, 5, 8, 3, 6, 3, 3, 2, 3, 8, 2, 3, 3, 3, 3, 8, 3, 2, 3, 3, 2, 2, 3, 3], 96個
#       device='cuda:0')
#正解値： tensor(3082, device='cuda:0')
