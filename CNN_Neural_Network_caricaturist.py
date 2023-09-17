
import torch.nn as nn

class SimpleCNN_caricaturist(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN_caricaturist, self).__init__()

        # 第一個卷積層：3個輸入通道，32個輸出通道，3x3的卷積核，填充1個像素
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # 添加BN层
        
        self.relu = nn.ReLU()  # ReLU 激活函數
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化層

        # 第二個卷積層：32個輸入通道，32個輸出通道，3x3的卷積核，填充1個像素
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # 添加BN层
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)  # 添加BN层
        
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)  # 添加BN层
        
        
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(32)  # 添加BN层
        
        # 計算展平後的特徵維度
        self.flatten_dim = 32 * 16 * 16  # 因為經過5次最大池化，尺寸減半5次
        
        
        # 添加残差块
        self.residual_block2 = self.make_residual_block_2(32, 32)

        
        
        # 第一個全連接層：將展平後的特徵映射，輸出256個特徵
        # 256 是你想要設計的全連接層的輸出維度。這個數字通常是根據你的任務需求和模型複雜度來選擇的。
        self.fc1 = nn.Linear(self.flatten_dim, 256)

        # 第二個全連接層：輸出 num_classes 個特徵，對應不同的畫家類別
        self.fc2 = nn.Linear(256, num_classes)
        
    
    def make_residual_block_2(self, in_channels, out_channels):
        # 创建一个包含两个卷积层的序列（Sequential）
        # 第一个卷积层：输入通道数为 in_channels，输出通道数为 out_channels，卷积核大小为 3x3，填充为 1（保持输入尺寸不变）
        # 激活函数：ReLU
        # 第二个卷积层：输入通道数为 out_channels（与第一个卷积层的输出通道数相同），输出通道数为 out_channels，卷积核大小为 3x3，填充为 1
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1,stride=2),# 设置步幅为 2
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,stride=2)# 设置步幅为 2
        )


        
    def forward(self, x):

        
        # 第一個卷積層、激活函數、最大池化
        x = self.conv1(x)
        x = self.bn1(x)  # 在卷积层后添加BN
        x = self.relu(x)
        x = self.pool(x)
        
        residual2 = self.residual_block2(x)
        
        # 第二個卷積層、激活函數、最大池化
        x = self.conv2(x)
        x = self.bn2(x)  # 在卷积层后添加BN
        x = self.relu(x)
        x = self.pool(x)


        x = self.conv3(x)
        x = self.bn3(x)  # 在卷积层后添加BN
        x = self.relu(x)
        x = self.pool(x)
        
        x = x + residual2
        
        # 从第四层到第五层的残差连接
        residual2 = self.residual_block2(x)
        
        x = self.conv4(x)
        x = self.bn4(x)  # 在卷积层后添加BN
        x = self.relu(x)
        x = self.pool(x)
        

        x = self.conv5(x)
        x = self.bn5(x)  # 在卷积层后添加BN
        x = self.relu(x)
        x = self.pool(x)
        
        x = x + residual2
             
        # 展平特徵映射，準備進入全連接層
        x = x.view(x.size(0), -1)
        
        # 第一個全連接層、激活函數
        x = self.fc1(x)
        x = self.relu(x)
        # 第二個全連接層，輸出最終分類結果
        x = self.fc2(x)
        return x
    
