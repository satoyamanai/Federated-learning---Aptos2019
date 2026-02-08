import requests
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import os
import time
import json
import traceback
from typing import Dict, Tuple, List
import random
from PIL import Image

# FIXED: 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

set_seed(42)

# ==================== 配置 ====================

CLIENT_ID = 4  
SERVER_URL = "http://localhost:5002"
DATASET_PATH = "/Users/yamanaisato/Desktop/Aptos2019"

# ==================== 简化模型（与服务器匹配） ====================

class SimpleDiabeticRetinopathyModel(nn.Module):
    """与服务器完全一致的模型"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 5)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==================== 数据集类（已修复） ====================
class APTOSDataset(Dataset):
    """加载和处理APTOS图像 - 使用PIL加载图像"""
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.image_paths = dataframe['filename'].tolist()
        self.labels = dataframe['diagnosis'].tolist()
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # FIXED: 使用PIL加载图像
            img = Image.open(img_path).convert('RGB')
            
            # FIXED: 应用transform（transform应处理PIL Image）
            if self.transform:
                img = self.transform(img)
            
            return img, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"处理图像时出错 {img_path}: {e}")
            # 返回空白图像作为容错
            blank_img = torch.zeros((3, 224, 224), dtype=torch.float32)
            return blank_img, torch.tensor(label, dtype=torch.long)

# ==================== 联邦学习客户端 ====================
class FederatedClient:
    def __init__(self, client_id: int, num_local_epochs: int = 2):
        self.client_id = client_id
        self.server_url = SERVER_URL
        self.num_local_epochs = num_local_epochs
        self.current_round = 0  # 跟踪当前轮次
        
        # 设备配置
        self.device = self._get_device()
        print(f"🖥️  客户端 {client_id} 使用设备: {self.device}")
        
        # 初始化模型
        self.model = SimpleDiabeticRetinopathyModel().to(self.device)
        
        # 加载数据
        self.train_loader, self.data_info = self._prepare_local_data()
        
        print(f"✅ 客户端 {client_id} 初始化完成")
        print(f"   数据量: {self.data_info['total_samples']} 图像")
        print(f"   类别分布: {self.data_info['class_distribution']}")
        print(f"   模型参数: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _get_device(self) -> torch.device:
        """获取最佳计算设备"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")
    
    def _prepare_local_data(self) -> Tuple[DataLoader, Dict]:
        """准备本地数据，模拟非IID分布"""
        if not os.path.exists(DATASET_PATH):
            raise FileNotFoundError(f"数据集路径不存在: {DATASET_PATH}")
        
        train_csv_path = os.path.join(DATASET_PATH, "train.csv")
        if not os.path.exists(train_csv_path):
            raise FileNotFoundError(f"CSV文件不存在: {train_csv_path}")
        
        # 加载数据
        train_df = pd.read_csv(train_csv_path)
        
        # 添加完整文件路径
        train_df["filename"] = train_df["id_code"].apply(
            lambda x: os.path.join(DATASET_PATH, "train_images", f"{x}.png")
        )
        
        # 检查文件是否存在
        existing_files = []
        for _, row in train_df.iterrows():
            if os.path.exists(row['filename']):
                existing_files.append(row)
        
        if not existing_files:
            raise FileNotFoundError("找不到任何图像文件")
        
        train_df = pd.DataFrame(existing_files)
        print(f"📊 找到 {len(train_df)} 个有效图像文件")
        
        # 模拟非IID分布
        if self.client_id == 1:
            condition = train_df['diagnosis'].isin([0, 1])
            subset = train_df[condition]
            if len(subset) > 100:
                client_df = subset.sample(n=100, random_state=42)
            else:
                client_df = subset
        elif self.client_id == 2:
            condition = train_df['diagnosis'] == 2
            subset = train_df[condition]
            if len(subset) > 100:
                client_df = subset.sample(n=100, random_state=43)
            else:
                client_df = subset
        elif self.client_id == 3:
            condition = train_df['diagnosis'] == 3
            subset = train_df[condition]
            if len(subset) > 80:
                client_df = subset.sample(n=80, random_state=44)
            else:
                client_df = subset
        elif self.client_id == 4:
            condition = train_df['diagnosis'] == 4
            subset = train_df[condition]
            if len(subset) > 60:
                client_df = subset.sample(n=60, random_state=45)
            else:
                client_df = subset
        else:
            client_df = train_df.sample(n=100, random_state=46)
        
        print(f"📁 客户端 {self.client_id} 分配到 {len(client_df)} 张图像")
        
        # FIXED: 正确的transform顺序
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 创建数据集
        dataset = APTOSDataset(client_df, transform=transform)
        
        batch_size = min(8, len(dataset))
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        # 统计信息
        class_dist = dict(client_df['diagnosis'].value_counts().sort_index())
        data_info = {
            'total_samples': len(dataset),
            'class_distribution': class_dist,
            'batch_size': batch_size,
            'num_batches': len(data_loader)
        }
        
        return data_loader, data_info
    
    def download_global_model(self) -> bool:
        """从服务器下载全局模型"""
        print(f"⬇️  客户端 {self.client_id} 正在下载全局模型...")
        
        try:
            # 尝试从get_model端点下载
            response = requests.get(
                f"{self.server_url}/get_model",
                timeout=30,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('success', False):
                    print(f"📥 收到服务器模型，轮次: {data.get('round', 0)}")
                    
                    # 检查轮次是否更新
                    server_round = data.get('round', 0)
                    if server_round != self.current_round:
                        print(f"🔄 更新客户端轮次: {self.current_round} -> {server_round}")
                        self.current_round = server_round
                    
                    # 反序列化模型权重
                    server_weights = data['model']
                    
                    # 转换为torch tensor
                    state_dict = {}
                    for key, value in server_weights.items():
                        tensor = torch.tensor(value, dtype=torch.float32)
                        state_dict[key] = tensor
                    
                    # 加载权重
                    self.model.load_state_dict(state_dict, strict=True)
                    self.model.to(self.device)
                    
                    print(f"✅ 成功加载全局模型")
                    print(f"   模型键值数量: {len(state_dict)}")
                    
                    return True
                else:
                    error_msg = data.get('error', '未知错误')
                    print(f"❌ 服务器返回错误: {error_msg}")
                    return False
            else:
                print(f"❌ 服务器响应错误: {response.status_code}")
                print(f"响应内容: {response.text[:200]}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"❌ 网络错误: {e}")
            return False
        except Exception as e:
            print(f"❌ 加载模型失败: {e}")
            traceback.print_exc()
            return False
    
    def local_train(self) -> Tuple[float, float]:
        """本地训练模型"""
        print(f"🎯 客户端 {self.client_id} 开始本地训练...")
        
        # 切换到训练模式
        self.model.train()
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        
        # 训练统计
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for epoch in range(self.num_local_epochs):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0
            
            for batch_idx, (images, labels) in enumerate(self.train_loader):
                # 检查数据类型
                if not isinstance(images, torch.Tensor):
                    print(f"⚠️ 警告: images类型异常: {type(images)}")
                    continue
                
                # 移到设备
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 统计
                batch_size = images.size(0)
                epoch_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs, 1)
                epoch_correct += (predicted == labels).sum().item()
                epoch_samples += batch_size
                
                # 每2个batch显示一次进度
                if batch_idx % 2 == 0 or batch_idx == len(self.train_loader) - 1:
                    batch_acc = (predicted == labels).sum().item() / max(1, batch_size)
                    print(f"   批次 {batch_idx}/{len(self.train_loader)}: "
                          f"Loss={loss.item():.4f}, Acc={batch_acc:.4f}")
            
            # 本轮统计
            epoch_avg_loss = epoch_loss / max(1, epoch_samples)
            epoch_accuracy = epoch_correct / max(1, epoch_samples)
            
            print(f"   Epoch {epoch+1}/{self.num_local_epochs}: "
                  f"Loss={epoch_avg_loss:.4f}, Acc={epoch_accuracy:.4f}")
            
            total_loss += epoch_loss
            total_correct += epoch_correct
            total_samples += epoch_samples
        
        # 总体统计
        avg_loss = total_loss / max(1, total_samples)
        avg_accuracy = total_correct / max(1, total_samples)
        
        print(f"✅ 客户端 {self.client_id} 训练完成:")
        print(f"   平均Loss: {avg_loss:.4f}")
        print(f"   平均准确率: {avg_accuracy:.4f}")
        
        return avg_loss, avg_accuracy
    
    def send_update(self) -> bool:
        """发送本地更新到服务器"""
        print(f"⬆️  客户端 {self.client_id} 正在发送更新...")
        
        try:
            # 获取模型权重
            self.model.to('cpu')
            model_state = self.model.state_dict()
            
            # 打印模型键值用于调试
            print("🔍 模型权重键值:")
            key_list = list(model_state.keys())
            for i, key in enumerate(key_list[:3]):
                shape = list(model_state[key].shape)
                print(f"  {i+1}. {key}: {shape}")
            if len(key_list) > 3:
                print(f"  ... 还有 {len(key_list)-3} 个键")
            
            # 序列化为可JSON传输的格式
            serializable = {}
            for key, tensor in model_state.items():
                # 确保是float32并转换为列表
                if tensor.is_floating_point():
                    tensor = tensor.to(torch.float32)
                serializable[key] = tensor.cpu().numpy().tolist()
            
            # 准备发送数据
            data = {
                'client_id': str(self.client_id),
                'model': serializable,
                'data_size': self.data_info['total_samples'],
                'current_round': self.current_round,  # 发送当前轮次
                'timestamp': time.time()
            }
            
            print(f"📤 发送数据大小: {len(str(data))} 字符")
            print(f"📤 模型键值数量: {len(serializable)}")
            
            # 发送到服务器
            response = requests.post(
                f"{self.server_url}/send_update",
                json=data,
                timeout=60,
                headers={'Content-Type': 'application/json'}
            )
            
            print(f"📥 服务器响应状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"📥 服务器响应: {result}")
                
                if result.get('success', False):
                    message = result.get('message', '更新成功')
                    print(f"✅ {message}")
                    
                    # 更新轮次信息
                    if result.get('round_completed', False):
                        new_round = result.get('new_round', self.current_round + 1)
                        print(f"🎉 轮次 {new_round} 聚合完成!")
                        self.current_round = new_round
                    
                    return True
                else:
                    error_msg = result.get('error', '未知错误')
                    print(f"❌ 服务器处理失败: {error_msg}")
                    return False
            else:
                print(f"❌ 服务器响应错误: {response.status_code}")
                print(f"📥 响应内容: {response.text[:200]}")
                return False
            
        except requests.exceptions.RequestException as e:
            print(f"❌ 网络错误: {e}")
            return False
        except Exception as e:
            print(f"❌ 发送更新失败: {e}")
            traceback.print_exc()
            return False
        finally:
            # 确保模型回到正确的设备
            self.model.to(self.device)
    
    def participate(self, num_rounds=3):
        """参与联邦学习过程"""
        print("=" * 60)
        print(f"🤖 联邦学习客户端 {self.client_id}")
        print(f"   服务器: {self.server_url}")
        print(f"   参与轮次: {num_rounds}")
        print("=" * 60)
        
        for round_num in range(num_rounds):
            print(f"\n{'='*40}")
            print(f"🔄 第 {round_num + 1}/{num_rounds} 轮")
            print(f"{'='*40}")
            
            # 1. 获取全局模型
            print(f"🔽 步骤1: 下载全局模型")
            if not self.download_global_model():
                print("⏳ 下载失败，等待10秒后重试...")
                time.sleep(10)
                if not self.download_global_model():
                    print("❌ 再次下载失败，跳过本轮")
                    continue
            
            # 2. 本地训练
            print(f"🎯 步骤2: 本地训练")
            try:
                loss, accuracy = self.local_train()
                print(f"📊 训练结果 - Loss: {loss:.4f}, Acc: {accuracy:.4f}")
            except Exception as e:
                print(f"❌ 训练失败: {e}")
                traceback.print_exc()
                continue
            
            # 3. 发送更新
            print(f"⬆️  步骤3: 发送更新")
            max_retries = 3
            for retry in range(max_retries):
                if self.send_update():
                    break
                elif retry < max_retries - 1:
                    wait_time = 5 * (retry + 1)
                    print(f"⏳ 发送失败，等待{wait_time}秒后重试...")
                    time.sleep(wait_time)
                else:
                    print("❌ 发送更新失败，跳过本轮")
            
            # 4. 等待下一轮
            if round_num < num_rounds - 1:
                wait_time = 10
                print(f"\n⏳ 等待 {wait_time} 秒进入下一轮...")
                time.sleep(wait_time)
        
        print("\n" + "=" * 60)
        print(f"🏁 客户端 {self.client_id} 完成所有联邦学习轮次!")
        print("=" * 60)

# ==================== 辅助函数 ====================
def check_server_health() -> bool:
    """检查服务器状态"""
    try:
        print("🔍 检查服务器状态...")
        response = requests.get(f"{SERVER_URL}/", timeout=10)
        
        if response.status_code == 200:
            server_info = response.json()
            print(f"✅ 服务器正常运行")
            print(f"   状态: {server_info.get('status', 'unknown')}")
            print(f"   当前轮次: {server_info.get('round', 0)}")
            print(f"   运行时间: {server_info.get('uptime_seconds', 0)} 秒")
            return True
        else:
            print(f"❌ 服务器响应异常: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到服务器")
        print("💡 请确保服务器已启动: python server.py")
        return False
    except Exception as e:
        print(f"❌ 检查服务器时出错: {e}")
        return False

def check_dataset() -> bool:
    """检查数据集是否存在"""
    print("🔍 检查数据集...")
    
    required_files = [
        ("train.csv", os.path.join(DATASET_PATH, "train.csv")),
        ("train_images目录", os.path.join(DATASET_PATH, "train_images"))
    ]
    
    all_ok = True
    for file_name, file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_name}: 存在")
        else:
            print(f"❌ {file_name}: 不存在 ({file_path})")
            all_ok = False
    
    if all_ok:
        try:
            csv_path = os.path.join(DATASET_PATH, "train.csv")
            df = pd.read_csv(csv_path)
            print(f"✅ CSV文件有效，包含 {len(df)} 行")
            print(f"   类别分布:")
            print(df['diagnosis'].value_counts().sort_index())
            return True
        except Exception as e:
            print(f"❌ 读取CSV失败: {e}")
            return False
    else:
        return False

# ==================== 主函数 ====================
def main():
    print("\n" + "=" * 60)
    print("联邦学习客户端 - APTOS糖尿病视网膜病变检测")
    print("=" * 60)
    print(f"客户端ID: {CLIENT_ID}")
    print(f"服务器: {SERVER_URL}")
    print(f"数据集路径: {DATASET_PATH}")
    print("=" * 60)
    
    # 1. 检查依赖
    print("\n📦 检查依赖...")
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        if torch.backends.mps.is_available():
            print("✅ Apple Silicon (MPS) 可用")
        elif torch.cuda.is_available():
            print("✅ CUDA 可用")
    except ImportError:
        print("❌ PyTorch未安装")
        print("💡 请运行: pip install torch torchvision")
        return
    
    # 2. 检查数据集
    print("\n📁 检查数据集...")
    if not check_dataset():
        print("❌ 数据集检查失败，请检查路径和文件")
        return
    
    # 3. 检查服务器
    print("\n🌐 检查服务器连接...")
    if not check_server_health():
        print("❌ 服务器检查失败")
        return
    
    # 4. 创建并运行客户端
    print("\n🚀 启动联邦学习客户端...")
    try:
        client = FederatedClient(
            client_id=CLIENT_ID,
            num_local_epochs=1  # 测试用1个epoch
        )
        
        # 参与联邦学习
        client.participate(num_rounds=2)  # 测试用2轮
        
    except KeyboardInterrupt:
        print("\n\n⚠️  客户端被用户中断")
    except Exception as e:
        print(f"\n❌ 客户端运行出错: {e}")
        traceback.print_exc()
        print("\n💡 调试建议:")
        print("1. 检查服务器是否运行: python server.py")
        print("2. 检查数据集路径是否正确")
        print("3. 查看详细错误信息")

if __name__ == '__main__':
    main()
