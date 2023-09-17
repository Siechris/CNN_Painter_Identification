import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from CNN_Neural_Network_caricaturist import SimpleCNN_caricaturist

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root  # 存儲資料集根目錄的路徑
        self.transform = transform  # 存儲資料轉換的函數
        self.images = []  # 存儲圖片路徑的列表
        self.labels = []  # 存儲標籤的列表
        
        # 列舉資料集根目錄中的子目錄名稱，每個子目錄名稱代表一個類別
        class_names = os.listdir(root)
        for class_name in class_names:
            class_dir = os.path.join(root, class_name)  # 獲取每個類別的目錄路徑
            if os.path.isdir(class_dir):  # 確保路徑是一個目錄
                for image_name in os.listdir(class_dir):  # 遍歷類別目錄中的圖片
                    image_path = os.path.join(class_dir, image_name)  # 獲取每個圖片的路徑
                    self.images.append(image_path)  # 將圖片路徑加入列表
                    class_name_to_label = {"minori_chigusa": 0, "スコッティ": 1,}  # 將類別名稱映射到整數標籤
                    label = class_name_to_label[class_name]  # 使用映射獲取整數標籤
                    self.labels.append(label)  # 將標籤加入列表
                    
    def __len__(self):
        return len(self.images)  # 返回資料集中的圖片數量

    def __getitem__(self, idx):
        image_path = self.images[idx]  # 獲取索引為 idx 的圖片路徑
        image = Image.open(image_path).convert('RGB')  # 讀取並轉換圖片格式為 RGB
        label = self.labels[idx]  # 獲取索引為 idx 的圖片標籤
        if self.transform:
            image = self.transform(image)  # 使用指定的轉換函數轉換圖片
        return image, label  # 返回轉換後的圖片和標籤



# 超參數
batch_size = 25 #一次訓練多少
num_epochs = 100
learning_rate = 0.0001


# 設定資料轉換
data_transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 將圖片調整為512x512大小
    transforms.ToTensor()  # 轉換為Tensor
])

# 載入訓練資料集
train_dataset = CustomDataset(root='./caricaturist', transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# 如果你的資料夾中的圖片數量不足以湊到一個完整的批次大小（例如25），
# 在最後一個批次中可能會有少於25張圖片。這在實際訓練中是正常的情況，不必過於擔心。

# 在PyTorch的DataLoader中，你可以使用參數 drop_last=True 來避免最後一個批次不足25的問題。
# 這將會在最後一個批次中，如果圖片數量不足25，則不使用該批次。這樣做可能會浪費少量資料，
# 但可以確保每個批次都具有相同的大小。


# 初始化模型
num_classes = 2  # 只有一個類別
model = SimpleCNN_caricaturist(num_classes=num_classes)

# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
# 定義優化器，同時設置 正則化weight_decay 參數weight_decay=0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.001)


# 訓練模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
model.to(device)



for epoch in range(num_epochs):
    
    # 這個語句將模型設置為訓練模式，這將影響某些層（例如，Dropout和BatchNormalization），
    # 使其在訓練模式下工作。
    model.train()
    running_loss = 0.0
    
    # 這個嵌套迴圈遍歷每個 batch 的訓練數據。train_loader 是一個用於批次式載入數據的 DataLoader，
    # inputs 是一批圖片，labels 是相應的標籤。
    for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 清除優化器的梯度，準備計算新的梯度
        optimizer.zero_grad()
        # 將輸入數據通過模型進行前向傳遞，得到模型的預測輸出。

        outputs = model(inputs)
        # 計算模型預測輸出與實際標籤之間的損失。
        loss = criterion(outputs, labels)
        # 計算損失的梯度，用於反向傳播。
        loss.backward()
        
               
        # 將當前 batch 的損失值添加到 running_loss 中，以便在 epoch 結束時計算平均損失。    
        running_loss += loss.item()
        
        # # 使用梯度裁剪，
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新模型參數，使損失最小化
        optimizer.step()
        

    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_loader)}')
    if (epoch + 1) % 5 == 0:

        # Save the model
        model_save_path = f'./CNN_caricaturist_model/simple_cnn_model_epoch_{epoch+1}.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')


print('Training finished!')




