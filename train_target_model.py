import torch #NumPY等を備えたTensorライブラリ
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader #Dataloader等のユーティリティ関数
import torch.nn.functional as F
from models import  MNIST_target_net #models内の定義


#python 〇〇.pyとして実行されているかどうか判定（importされても動かない）
if __name__ == "__main__": #__name__を使うと、モジュール名が文字列で入る（例：math.__name__なら"math"）、しかしコマンドラインから実行すると"__main__"が入る
    use_cuda = True
    image_nc = 1
    batch_size = 256

    # Define what device we are using
    print("CUDA Available: ", torch.cuda.is_available()) #GPUが利用可能か
    #コマンドラインから実行され、GPUが利用可能なら"cuda"、それ以外は"cpu"を「使用できるデバイス名」として変数に格納
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    #'./dataset'はデータセットが存在するディレクトリ　train=Trueならtrain-image-idx3-ubyteからデータセット作成、falseならt10k-images-idx3-ubyteで作成
    #transformは、PILイメージを取り込み、変換されたバージョンを返す（今回はtorchvision.transforms）　download=Trueはダウンロードする
    mnist_dataset = torchvision.datasets.MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    # mnist_datasetからサンプルを取得し、訓練データ（ミニバッチ）を作成　shuffle=シャッフルするかどうか　num_workers=並列実行数
    train_dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # training the target model
    target_model = MNIST_target_net().to(device)
    target_model.train()
    opt_model = torch.optim.Adam(target_model.parameters(), lr=0.001) #optim=最適化、lrは学習率(0.1以下なら収束)
    epochs = 40
    for epoch in range(epochs):
        loss_epoch = 0
        if epoch == 20:
            opt_model = torch.optim.Adam(target_model.parameters(), lr=0.0001) #lrは学習率(0.1以下なら収束)
        #enumerate()のカッコ内にリスト(train_dataloader)等を指定、iとdataに番号(0,1,2,...)と要素を格納、番号の開始値は0に指定
        for i, data in enumerate(train_dataloader, 0):
            train_imgs, train_labels = data
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            logits_model = target_model(train_imgs)
            loss_model = F.cross_entropy(logits_model, train_labels) #クロスエントロピー誤差
            loss_epoch += loss_model
            opt_model.zero_grad()
            loss_model.backward()
            opt_model.step()

        print('loss in epoch %d: %f' % (epoch, loss_epoch.item())) #損失を出力

    # save model
    targeted_model_file_name = './MNIST_target_model.pth'
    torch.save(target_model.state_dict(), targeted_model_file_name)
    target_model.eval()

    # MNIST test dataset
    mnist_dataset_test = torchvision.datasets.MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)
    test_dataloader = DataLoader(mnist_dataset_test, batch_size=batch_size, shuffle=True, num_workers=1)
    num_correct = 0
    for i, data in enumerate(test_dataloader, 0): #enumerate()のカッコ内にリスト等を指定、iとdataに番号(0,1,2,...)と要素を入れる
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        pred_lab = torch.argmax(target_model(test_img), 1)
        num_correct += torch.sum(pred_lab==test_label,0)

    print('accuracy in testing set: %f\n'%(num_correct.item()/len(mnist_dataset_test))) #正確さを出力
