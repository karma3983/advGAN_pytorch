
#https://qiita.com/mathlive/items/8e1f9a8467fff8dfd03c
#https://qiita.com/poorko/items/c151ff4a827f114fe954
#2dmaxpool https://cvml-expertguide.net/terms/dl/layers/pooling-layer/max-pooling/#:~:text=%E6%9C%80%E5%A4%A7%E5%80%A4%E3%83%97%E3%83%BC%E3%83%AA%E3%83%B3%E3%82%B0(Max%20Pooling%20)%E3%81%A8%E3%81%AF%EF%BC%8CCNN(,%E6%AE%8B%E3%81%99%E3%83%97%E3%83%BC%E3%83%AA%E3%83%B3%E3%82%B0%E5%87%A6%E7%90%86%E3%81%A7%E3%81%82%E3%82%8B%EF%BC%8E

#nnはパラメータを持つ層、Fはパラメータを持たない層がそれぞれ入っているモジュール
import torch.nn as nn
import torch.nn.functional as F #conv2d、活性化関数(relu等)、dropout、cross_entropy等


# Target Model definition
class MNIST_target_net(nn.Module):
    #インスタンス（○○ = MNIST_target_net）を生成した際に、1番最初に呼び出される関数（コンストラクタ）
    def __init__(self): #self：メソッド(def)を跨いでも同じ変数扱いとなる
        super(MNIST_target_net, self).__init__() #親(MNIST_target_net)のコンストラクタ(__init__)を呼び出す
        #2次元畳み込み　入力チャネル=1、出力チャネル=32、カーネルサイズ3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)

        #入力データに線形変換　入力サイズ(1次元) = 64チャネル * 4height * 4width、出力サイズ：200
        self.fc1 = nn.Linear(64*4*4, 200)
        self.fc2 = nn.Linear(200, 200)
        self.logits = nn.Linear(200, 10)

    #引数としてデータ（x）を受け取り、出力層の値を出す(順伝播の流れ)
    def forward(self, x):
        x = F.relu(self.conv1(x)) #nn.functional.relu：データが0より大きければその値を出力し、0より小さければ0
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2) #x:入力値　カーネルサイズ2　2D最大値プーリング処理（カーネル内の最大値を出力する）
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        #-1を入れることで、2つ目の値にサイズ数を調整(Tensorの要素数に合わせる必要あり)
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x)) #self.fc1 = nn.Linear(64*4*4, 200)
        #一定割合(50%)のTensor要素を不活性化(0に)させながら学習を行い、過学習を防ぐ
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x)) #self.fc2 = nn.Linear(200, 200)
        x = self.logits(x) #self.logits = nn.Linear(200, 10)
        return x


class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        # MNIST: 1*28*28
        model = [
            nn.Conv2d(image_nc, 8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            # 8*13*13
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # 16*5*5
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
            # 32*1*1
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x).squeeze()
        return output


class Generator(nn.Module):
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(Generator, self).__init__()

        encoder_lis = [
            # MNIST:1*28*28
            nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # 8*26*26
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # 16*12*12
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32*5*5
        ]

        bottle_neck_lis = [ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),]

        decoder_lis = [
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # state size. 16 x 11 x 11
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # state size. 8 x 23 x 23
            nn.ConvTranspose2d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Tanh()
            # state size. image_nc x 28 x 28
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x


# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
