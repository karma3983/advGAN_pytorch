import torch.nn as nn
import torch
import numpy as np
import models
import torch.nn.functional as F
import torchvision
import os

models_path = './models/'

#netGとnetDで呼ばれる重みの初期化
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__ #クラス名を取得
    if classname.find('Conv') != -1: #Convという名前の場合
        #Tensor(m.weight.data)を正規分布(重み、平均0.0、標準偏差0.02)で初期化
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1: #BatchNormという名前の場合
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        #Tensor(m.biad.data)を定数0で初期化するー＞例：tensor([0,0,0,0,0,0,0,0,0])
        nn.init.constant_(m.bias.data, 0)

#main.pyで使用
class AdvGAN_Attack:
    def __init__(self,
                 device, #CUDA
                 model, #MNIST_target_net
                 model_num_labels, #10
                 image_nc, #1
                 box_min, #0
                 box_max): #1
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max

        self.gen_input_nc = image_nc #1
        self.netG = models.Generator(self.gen_input_nc, image_nc).to(device) #modelsファイル device='cuda:0'付き
        self.netDisc = models.Discriminator(image_nc).to(device) #modelsファイル device='cuda:0'付き

        # initialize all weights 重み初期化
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers 最適化初期化
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.001)

        #models_path = './models/'、models_pathが存在しないなら追加
        if not os.path.exists(models_path):
            os.makedirs(models_path)

    def train_batch(self, x, labels): #images, labels
        # optimize D ディスクリミネーター最適化
        for i in range(1):
            perturbation = self.netG(x) #self.netG = models.Generator(self.gen_input_nc, image_nc).to(device)

            # add a clipping trick
            adv_images = torch.clamp(perturbation, -0.3, 0.3) + x #上限0.3、下限-0.3にTensorをクランプ(元々範囲内なら効果なし)
            adv_images = torch.clamp(adv_images, self.box_min, self.box_max) #上限1、下限0にTensorをクランプ(元々範囲内なら効果なし)

            self.optimizer_D.zero_grad() #52行の勾配を0に
            pred_real = self.netDisc(x) #self.netDisc = models.Discriminator(image_nc).to(device)
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward() #誤差逆伝播(この際勾配が溜まる)

            pred_fake = self.netDisc(adv_images.detach())  #self.netDisc = models.Discriminator(image_nc).to(device)
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward() #誤差逆伝播(この際勾配が溜まる)
            loss_D_GAN = loss_D_fake + loss_D_real #★★★論文のLGANに相当するもの★★★
            self.optimizer_D.step() #パラメータをモデルに反映

        # optimize G ジェネレータ最適化
        for i in range(1):
            self.optimizer_G.zero_grad() #50行の勾配を0に

            # cal G's loss in GAN
            pred_fake = self.netDisc(adv_images)
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device)) #★★★Generatorの損失★★★(多分)
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            C = 0.1
            loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
            # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

            # cal adv loss
            logits_model = self.model(adv_images)
            probs_model = F.softmax(logits_model, dim=1)
            onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels] #torch.eye：対角線上の要素が全て1、他は全て0の行列

            # C&W loss function
            real = torch.sum(onehot_labels * probs_model, dim=1) #分類できたデータの総和？
            other, _ = torch.max((1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1) #最も分類できなかったデータ*10000？
            zeros = torch.zeros_like(other)
            loss_adv = torch.max(real - other, zeros) #どちらが大きいか
            loss_adv = torch.sum(loss_adv)

            # maximize cross_entropy loss
            # loss_adv = -F.mse_loss(logits_model, onehot_labels)
            # loss_adv = - F.cross_entropy(logits_model, labels)

            adv_lambda = 10
            pert_lambda = 1
            #{10 * loss_adv = torch.sum(loss_adv)} + {1 * loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))}
            loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()

    #main.pyで実行される、DataLoader(mnist_dataset, batch_size=128, shuffle=True, num_workers=1)、エポック数60
    def train(self, train_dataloader, epochs):
        for epoch in range(1, epochs+1):

            if epoch == 50:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.0001)
            if epoch == 80:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=0.00001)
                self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                                    lr=0.00001)
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            #enumerate()のカッコ内にリスト(train_dataloader)等を指定、iとdataに番号(0,1,2,...)と要素を格納、開始値は0に指定
            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = \
                    self.train_batch(images, labels)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch
                
                #print("番号：",i)
                #print("画像",images)
                #print("ラベル：",labels)
                #print("loss_D_batch：",loss_D_batch)
                #print("loss_G_fake_batch：",loss_G_fake_batch)
                #print("loss_perturb_batch：",loss_perturb_batch)
                #print("loss_adv_batch：",loss_adv_batch)
                #print("-------------------------------------------")

            # print statistics 統計を表示
            num_batch = len(train_dataloader) #必ず469
#            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
#             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
#                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
#                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))

            #print("train_dataloaderの長さ：",len(train_dataloader)) #469を出す方法
            #loss_D:0に近くなる、loss_G_fake:1に近くなる、loss_perturb:5~6くらい？、loss_adv_sum:0.1くらい？
            print("epoch %d:\nloss_D: %.6f, loss_G_fake: %.6f,\
             \nloss_perturb: %.6f, loss_adv: %.6f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))

            # save generator
            if epoch%20==0:
                #models + netG_epoch_ + 20 or 40 or 60 + .pth
                netG_file_name = models_path + 'netG_epoch_' + str(epoch) + '.pth' #models_path = './models/'
                torch.save(self.netG.state_dict(), netG_file_name)

                
            netG_file_name = models_path + 'netG_epoch_' + str(epoch) + '.pth' #models_path = './models/' #検証用
            torch.save(self.netG.state_dict(), netG_file_name) #検証用
