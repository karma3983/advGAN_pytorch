#使用モジュール
import torch #NumPY等を備えたTensorライブラリ
import torchvision.datasets #Pytorchの画像データセット(今回はMNIST)
import torchvision.transforms as transforms #ToTensor(入力：PILImage　出力：Tensor)等のtransform
from torch.utils.data import DataLoader #Dataloader等のユーティリティ関数
import torch.nn.functional as F #conv2d、活性化関数(relu等)、dropout、cross_entropy等
from models import  MNIST_target_net #models内の定義
import time
#import matplotlib.pyplot as plt

#python 〇〇.pyとして実行されているかどうか判定（importされても動かない）
#__name__を使うと、モジュール名が文字列で入る（例：math.__name__なら"math"）、しかしコマンドラインから実行すると"__main__"が入る
if __name__ == "__main__":
    use_cuda = True
    image_nc = 1
    batch_size = 256

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available()) #GPUが利用可能か
    #コマンドラインから実行され、GPUが利用可能なら"cuda"、それ以外は"cpu"を「使用できるデバイス名」として変数に格納
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    print("使用デバイス：",device)

    #'./dataset'はデータセットが存在するディレクトリ
    #train=Trueならtrain-image-idx3-ubyteからデータセット作成、falseならt10k-images-idx3-ubyteで作成
    #transformは、PILイメージを取り込み、変換されたバージョンを返す（今回はtorchvision.transforms）　download=Trueはダウンロードする
    mnist_dataset = torchvision.datasets.MNIST('./dataset', 
                                               train=True, 
                                               transform=transforms.ToTensor(), 
                                               download=True)
    # mnist_datasetからサンプルを取得し、訓練データ（ミニバッチ）を作成　shuffle=実行する度ランダムにするかどうか　num_workers=並列実行数
    train_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # training the target model　対象モデルの学習
    target_model = MNIST_target_net().to(device)
    target_model.train() #model.train()：ネットワークを学習モードに(DropoutとBatchNormに影響するフラグ)
    opt_model = torch.optim.Adam(target_model.parameters(), lr=0.001) #optim=最適化、lrは学習率(0.1以下なら収束)
    epochs = 40 #本来は40------------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(epochs): #0から40
        #start = time.time()
        loss_epoch = 0
        if epoch == 20:
            opt_model = torch.optim.Adam(target_model.parameters(), lr=0.0001) #lrは学習率(0.1以下なら収束)
        #enumerate()のカッコ内にリスト(train_dataloader)等を指定、iとdataに番号(0,1,2,...)と要素を格納、番号の開始値は0に指定
        for i, data in enumerate(train_dataloader, 0):
            train_imgs, train_labels = data #train_dataloaderの要素を代入
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device) # device='cuda:0'が付くだけ
            logits_model = target_model(train_imgs)
            loss_model = F.cross_entropy(logits_model, train_labels) #クロスエントロピー誤差（logits_modelは予測値、train_labelsは正解ラベル）
            loss_epoch += loss_model
            opt_model.zero_grad() #勾配をゼロクリアする(パラメータ更新)
            loss_model.backward() #誤差逆伝播(この際勾配が溜まる)
            opt_model.step() #パラメータをモデルに反映
            
            #print("「学習」番号：",i)
            #print("画像：",train_imgs) #ずーっと0のTensor
            #print("ラベル：",train_labels) #256個*234、最後96個->60000個
            #print("ロジットモデル：",logits_model) #長い
            #print("損失モデル：",loss_model)
            #print("合計損失：", loss_epoch)
            #print("損失エポック.item：", loss_epoch.item()) #数字のみ
            #print("----------------------------------")

        print('loss in epoch %d: %f' % (epoch, loss_epoch.item())) #損失を出力
        #print ('時間：{} 秒'.format(time.time()-start))

    # save model
    targeted_model_file_name = './MNIST_target_model.pth'
    torch.save(target_model.state_dict(), targeted_model_file_name) #target_model.state_dict()を保存('./MNIST_target_model.pth'に)
    target_model.eval() #model.eval()：self.train(False)を返す、ネットワークを推論モードに、DropoutやBatchNormの on/offの切替？

    # MNIST test dataset
    #'./dataset'はデータセットが存在するディレクトリ　train=Trueならtrain-image-idx3-ubyteからデータセット作成、falseならt10k-images-idx3-ubyteで作成
    #transformは、PILイメージを取り込み、変換されたバージョンを返す（今回はtorchvision.transforms）　download=Trueはダウンロードする
    mnist_dataset_test = torchvision.datasets.MNIST('./dataset', 
                                                    train=False, 
                                                    transform=transforms.ToTensor(), 
                                                    download=True)
    # mnist_datasetからサンプルを取得し、訓練データ（ミニバッチ）を作成　shuffle=実行する度ランダムにするかどうか　num_workers=並列実行数
    test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=True, num_workers=1)
    num_correct = 0
    #enumerate()のカッコ内にリスト(train_dataloader)等を指定、iとdataに番号(0,1,2,...)と要素を格納、番号の開始値は0に指定
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        #argmax：Tensor(PyTorchで必ずデータ構造)の全ての要素の内、最大のインデックスを返す
        pred_lab = torch.argmax(target_model(test_img), 1) #target_model = MNIST_target_net().to(device)　なお、dim(縮小する次元)=1
        num_correct += torch.sum(pred_lab==test_label,0) #予測値＝ラベル、pred_lab==test_labelがdimを保持しているか確認
        
        #print("「テスト」番号：",i)
        #print("画像：",test_img) #ずーっと0のTensor
        #print("ラベル：",test_label) #256個*39、最後16個->10000個
        #print("画像の最大インデックス（予測値）：",pred_lab) #256個*39、最後16個->10000個
        #print("一致数：",torch.sum(pred_lab==test_label,0)) #device='cuda:0'付き
        #print("合計一致数：",num_correct) #device='cuda:0'付き
        #print("一致数：",torch.sum(pred_lab==test_label,0).item())
        #print("合計一致数：",num_correct.item())
        #print("----------------------------------")

    print('accuracy in testing set: %f\n'%(num_correct.item()/len(mnist_dataset_test))) #正確さを出力
    
    # ------------------------------------------------------------------------
    
    mnist_dataset_test = torchvision.datasets.MNIST('./dataset', 
                                                    train=True, 
                                                    transform=transforms.ToTensor(), 
                                                    download=True)
    # mnist_datasetからサンプルを取得し、訓練データ（ミニバッチ）を作成　shuffle=シャッフルするかどうか　num_workers=並列実行数
    test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=True, num_workers=1)
    num_correct = 0
    #enumerate()のカッコ内にリスト(train_dataloader)等を指定、iとdataに番号(0,1,2,...)と要素を格納、番号の開始値は0に指定
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        #argmax：Tensor(PyTorchで必ずデータ構造)の全ての要素の内、最大のインデックスを返す
        pred_lab = torch.argmax(target_model(test_img), 1) #target_model = MNIST_target_net().to(device)　なお、dim(縮小する次元)=1
        num_correct += torch.sum(pred_lab==test_label,0) #予測値＝ラベル、pred_lab==test_labelがdimを保持しているか確認
        
        #print("「学習」番号：",i)
        #print("画像：",test_img) #ずーっと0のTensor
        #print("ラベル：",test_label) #256個*234、最後96個->60000個
        #print("画像の最大インデックス（予測値）：",pred_lab) #256個*234、最後96個->60000個
        #print("一致数：",torch.sum(pred_lab==test_label,0)) #device='cuda:0'付き
        #print("合計一致数：",num_correct) #device='cuda:0'付き
        #print("一致数：",torch.sum(pred_lab==test_label,0).item())
        #print("合計一致数：",num_correct.item())
        #print("----------------------------------")

    print('accuracy in trainning set: %f\n'%(num_correct.item()/len(mnist_dataset_test))) #正確さを出力
    
#「学習」番号： 234
#画像： tensor([[[[0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          ...,
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.]]],
#          省略   device='cuda:0')
#ラベル： tensor([3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
#        6, 0, 3, 4, 1, 4, 0, 7, 8, 7, 7, 9, 0, 4, 9, 4, 0, 5, 8, 5, 9, 8, 8, 4,
#        0, 7, 1, 3, 5, 3, 1, 6, 5, 3, 8, 7, 3, 1, 6, 8, 5, 9, 2, 2, 0, 9, 2, 4,
#        6, 7, 3, 1, 3, 6, 6, 2, 1, 2, 6, 0, 7, 8, 9, 2, 9, 5, 1, 8, 3, 5, 6, 8],
#       device='cuda:0')
#ロジットモデル： tensor([[-1.5067e+01, -2.5206e+00, -8.0277e+00,  1.9538e+01, -1.0110e+01,
#          8.5212e+00, -2.0763e+01, -5.2299e+00, -7.2228e+00,  1.5391e+00],
#        [-2.4181e+00, -4.2801e+00, -3.1173e+00, -6.4889e+00,  1.0085e+01],
#          省略  ],
#       device='cuda:0', grad_fn=<AddmmBackward0>)
#損失モデル： tensor(0.1335, device='cuda:0', grad_fn=<NllLossBackward0>)
#合計損失： tensor(10.8233, device='cuda:0', grad_fn=<AddBackward0>)
#損失エポック.item： 10.823270797729492
#----------------------------------
#loss in epoch 4: 10.823271
#accuracy in testing set: 0.981000
#
#accuracy in trainning set: 0.984350
#-----------------------------------------------------------------------------------------------------------------    
#
#「学習」番号： 234
#画像： tensor([[[[0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          ...,
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.],
#          [0., 0., 0.,  ..., 0., 0., 0.]]],
#          省略  device='cuda:0')
#ラベル： tensor([8, 6, 6, 3, 9, 5, 1, 4, 2, 2, 2, 3, 1, 2, 9, 7, 0, 8, 1, 8, 7, 9, 8, 1,
#        4, 3, 0, 1, 7, 2, 2, 4, 5, 4, 5, 0, 2, 0, 5, 9, 3, 6, 6, 4, 9, 6, 4, 0,
#        4, 8, 9, 8, 4, 3, 8, 7, 1, 2, 9, 2, 5, 9, 7, 2, 1, 6, 2, 8, 3, 8, 1, 2,
#        4, 0, 4, 5, 7, 8, 9, 0, 3, 5, 7, 0, 6, 6, 1, 4, 1, 2, 9, 7, 8, 2, 2, 5],
#       device='cuda:0')
#画像の最大インデックス（予測値）： tensor([8, 6, 6, 3, 9, 5, 1, 4, 2, 2, 2, 3, 1, 2, 9, 7, 0, 8, 1, 8, 9, 9, 8, 1,
#        4, 3, 0, 1, 9, 2, 2, 4, 5, 4, 5, 0, 2, 0, 5, 9, 3, 5, 6, 4, 9, 6, 4, 0,
#        4, 8, 9, 8, 4, 3, 8, 7, 1, 2, 9, 2, 5, 9, 7, 2, 1, 6, 2, 8, 3, 8, 1, 2,
#        4, 0, 4, 5, 7, 8, 9, 0, 3, 5, 7, 0, 6, 6, 1, 4, 1, 2, 9, 7, 8, 2, 2, 5],
#       device='cuda:0')
#一致数： tensor(93, device='cuda:0')
#合計一致数： tensor(59136, device='cuda:0')
#一致数： 93
#合計一致数： 59136
#----------------------------------
#accuracy in trainning set: 0.985600
