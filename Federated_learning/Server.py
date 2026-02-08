# server.py - 联邦学习服务器
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import json
import os
import time
from datetime import datetime
from pathlib import Path
import traceback
import copy

# 导入配置
from config import config

app = Flask(__name__)

# 模型定义
class DiabeticRetinopathyModel(nn.Module):
    """与Kaggle项目匹配的简化CNN模型"""
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
            nn.Linear(256, config.get("model.num_classes", 5))
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# 服务器状态管理 
class ServerState:
    """管理服务器状态"""
    def __init__(self):
        self.global_model = DiabeticRetinopathyModel()
        self.client_updates = []  # 存储客户端更新
        self.client_ids = []      # 记录已提交的客户端
        self.round_num = 0
        self.start_time = time.time()
        
        # 创建保存目录
        self.save_path = Path(config.get("server.model_save_path", "./saved_models"))
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # 加载最新检查点（如果有）
        self.load_latest_checkpoint()
    
    def load_latest_checkpoint(self):
        """加载最新的模型检查点"""
        checkpoints = list(self.save_path.glob("global_model_round_*.pth"))
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            self.global_model.load_state_dict(torch.load(latest))
            self.round_num = int(latest.stem.split('_')[-1])
            print(f"✅ 加载检查点: {latest.name} (轮次 {self.round_num})")
    
    def save_checkpoint(self):
        """保存模型检查点"""
        filename = self.save_path / f"global_model_round_{self.round_num}.pth"
        torch.save(self.global_model.state_dict(), filename)
        
        # 同时保存为最新版本
        latest_path = self.save_path / "global_model_latest.pth"
        torch.save(self.global_model.state_dict(), latest_path)
        
        print(f"💾 保存检查点: {filename.name}")
    
    def add_client_update(self, client_id, update):
        """添加客户端更新"""
        if client_id not in self.client_ids:
            self.client_updates.append({
                'client_id': client_id,
                'weights': update,
                'timestamp': time.time()
            })
            self.client_ids.append(client_id)
            return True
        return False  # 客户端已提交
    
    def reset_round(self):
        """重置轮次状态"""
        self.client_updates = []
        self.client_ids = []
    
    def federated_average(self):
        """
        FIXED (Final Version): Correctly extracts 'weights' from the dictionary envelope.
        """

        if len(self.client_updates) < self.min_clients:
            return False

        print(f"🔄 Aggregating updates from {len(self.client_updates)} clients...")

        first_client_update_wrapper = self.client_updates[0]
        first_client_weights = first_client_update_wrapper['weights'] 
        
        avg_weights = copy.deepcopy(first_client_weights)


        for i in range(1, len(self.client_updates)):
            client_wrapper = self.client_updates[i]
            client_weights = client_wrapper['weights'] # <--- Extract weights here too
            
            for key in avg_weights:
                # Accumulate the tensors
                avg_weights[key] += client_weights[key]

        num_clients = len(self.client_updates)
        for key in avg_weights:
            avg_weights[key] = avg_weights[key].float() / num_clients

        self.global_model.load_state_dict(avg_weights)
        self.save_model()
        
        self.client_updates = []
        self.current_round += 1
        print(f"✅ Round {self.current_round} complete. Global model updated.")
        
        return True

# 初始化服务器状态
server_state = ServerState()

# Flask路由 
@app.route('/')
def home():
    """服务器状态页"""
    uptime = time.time() - server_state.start_time
    return jsonify({
        "status": "running",
        "round": server_state.round_num,
        "uptime_seconds": int(uptime),
        "clients_registered": len(server_state.client_ids),
        "model_info": {
            "name": "DiabeticRetinopathyModel",
            "input_size": config.get("model.input_size", 224),
            "num_classes": config.get("model.num_classes", 5)
        }
    })

@app.route('/get_model', methods=['GET'])
def get_global_model():
    """客户端获取当前全局模型"""
    try:
        model_state = server_state.global_model.state_dict()
        
        # 转换为可序列化的格式
        serializable = {}
        for key, tensor in model_state.items():
            serializable[key] = tensor.cpu().numpy().tolist()
        
        return jsonify({
            'success': True,
            'round': server_state.round_num,
            'model': serializable,
            'model_structure': str(server_state.global_model)
        })
    
    except Exception as e:
        print(f"❌ 获取模型失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/send_update', methods=['POST'])
def receive_update():
    """接收客户端模型更新"""
    try:
        data = request.json
        
        if not data or 'client_id' not in data or 'model' not in data:
            return jsonify({
                'success': False,
                'error': '缺少必要字段: client_id 或 model'
            }), 400
        
        client_id = str(data['client_id'])
        client_model = data['model']
        
        print(f"📥 收到客户端 {client_id} 的更新")
        
        # 添加到更新列表
        if not server_state.add_client_update(client_id, client_model):
            return jsonify({
                'success': False,
                'error': f'客户端 {client_id} 已提交过本轮更新'
            }), 400
        
        # 检查是否达到聚合阈值
        min_clients = config.get("server.min_clients", 2)
        if len(server_state.client_updates) >= min_clients:
            print(f"🎯 达到聚合条件 ({len(server_state.client_updates)}/{min_clients} 客户端)")
            
            # 执行联邦平均
            if server_state.federated_average():
                # 定期保存检查点
                checkpoint_interval = config.get("server.checkpoint_interval", 5)
                if server_state.round_num % checkpoint_interval == 0:
                    server_state.save_checkpoint()
                
                # 重置轮次状态
                server_state.reset_round()
                
                return jsonify({
                    'success': True,
                    'message': f'轮次 {server_state.round_num} 聚合完成',
                    'round_completed': True,
                    'new_round': server_state.round_num
                })
        
        return jsonify({
            'success': True,
            'message': f'更新已接收，等待更多客户端 ({len(server_state.client_updates)}/{min_clients})',
            'round_completed': False
        })
    
    except Exception as e:
        print(f"❌ 处理客户端更新失败: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'服务器错误: {str(e)}'
        }), 500

@app.route('/server_info', methods=['GET'])
def server_info():
    """获取详细服务器信息"""
    model_size = sum(p.numel() for p in server_state.global_model.parameters())
    
    return jsonify({
        "server_config": config.config,
        "current_round": server_state.round_num,
        "active_clients": len(server_state.client_ids),
        "pending_updates": len(server_state.client_updates),
        "model_statistics": {
            "total_parameters": model_size,
            "trainable_parameters": sum(p.numel() for p in server_state.global_model.parameters() if p.requires_grad)
        },
        "checkpoints": {
            "save_path": str(server_state.save_path),
            "latest_round": server_state.round_num
        }
    })

# 启动服务器
if __name__ == '__main__':
    host = config.get("server.host", "0.0.0.0")
    port = config.get("server.port", 5002)
    
    print("=" * 60)
    print("联邦学习服务器 - 糖尿病视网膜病变检测")
    print("=" * 60)
    print(f"📁 模型保存路径: {server_state.save_path}")
    print(f"🤖 模型参数数量: {sum(p.numel() for p in server_state.global_model.parameters()):,}")
    print(f"🔧 配置: 最少 {config.get('server.min_clients', 2)} 个客户端触发聚合")
    print(f"🌐 服务器地址: http://{host}:{port}")
    print("=" * 60)
    
    app.run(host=host, port=port, debug=False)
