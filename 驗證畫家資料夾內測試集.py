import torch
from torchvision import transforms
from PIL import Image
from CNN_Neural_Network_caricaturist import SimpleCNN_caricaturist

name_to_label = { 0:"minori_chigusa", 1:"スコッティ"}  # 將類別名稱映射到整數標籤


# 設定資料轉換
data_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 初始化模型
num_classes = 2  # 只有一個類別
model = SimpleCNN_caricaturist(num_classes=num_classes)

# 載入已經訓練好的模型參數
model.load_state_dict(torch.load('./CNN_caricaturist_model/simple_cnn_model_epoch_100.pth'))
model.eval()


import os
print("測試集驗證")

for i in range(len(name_to_label)):

    # 指定图像文件夹路径
    image_folder_path = "./caricaturist_test/"+str(name_to_label[i])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    # 初始化列表以存储每张图像的概率
    all_probabilities = []
    
    
    # 遍历文件夹中的所有图像
    for image_file in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, image_file)
    
        # 载入图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = data_transform(image).unsqueeze(0).to(device, dtype=torch.float32)
    
        # 使用模型进行预测
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].tolist()
    
        # 将概率添加到列表中
        all_probabilities.append(probabilities)
    
    print("======================================")
    
    # 计算平均概率
    avg_probabilities = [sum(p[i] for p in all_probabilities) / len(all_probabilities) for i in range(num_classes)]
    
    
    # 找到最大概率的索引
    max_probability_index = avg_probabilities.index(max(avg_probabilities))
    
    
    # 打印最有可能的类别
    print(f'最有可能的类别: {name_to_label[max_probability_index]}')
    print(f'整體概率: {avg_probabilities[max_probability_index]:.2%}')

print("======================================")
print("\n驗證畫師訓練出來的lora步數差異 ，stable diffusion大模型為 Bomain 64生成 ")

for i in range(1,11):
    image_folder_path = "./caricaturist_test/"+str(i)
    print("======================================")     
    print("步數 : "+str(i*500))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    # 初始化列表以存储每张图像的概率
    all_probabilities = []
    
    
    # 遍历文件夹中的所有图像
    for image_file in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, image_file)
    
        # 载入图像
        image = Image.open(image_path).convert('RGB')
        image_tensor = data_transform(image).unsqueeze(0).to(device, dtype=torch.float32)
    
        # 使用模型进行预测
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0].tolist()
    
        # 将概率添加到列表中
        all_probabilities.append(probabilities)
    
    
    # 计算平均概率
    avg_probabilities = [sum(p[i] for p in all_probabilities) / len(all_probabilities) for i in range(num_classes)]
    
    
    # 找到最大概率的索引
    max_probability_index = avg_probabilities.index(max(avg_probabilities))
    
    
    # 打印最有可能的类别
    print(f'最有可能的类别: {name_to_label[max_probability_index]}')
    print(f'整體概率: {avg_probabilities[max_probability_index]:.2%}')




