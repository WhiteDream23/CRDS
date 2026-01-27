import numpy as np
import torch
import matplotlib.pyplot as plt
from old_code.e2emodel import RingdownCNN
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
def evaluate_model(model_path, data_file, output_file="prediction_results.npz", device=None):
    """
    使用预训练模型对数据进行预测和评估

    参数:
        model_path: 预训练模型的路径
        data_file: 处理后的数据文件路径(.npz)
        output_file: 预测结果输出文件路径
        device: 使用的设备 ('cpu' 或 'cuda')
    """
    # 加载数据
    data = np.load(data_file)
    # input_sequences = data['data']
    # taus = data['filenames']
    input_sequences = data['data']
    taus = data['targets']
    tau_min, tau_max = 5, 25

    # 加载模型
    checkpoint = torch.load(model_path, map_location=device,weights_only=True)

    # 创建与训练时相同的模型实例
    model = RingdownCNN(dropout_rate=0.2)

    # 加载模型参数
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    all_predictions = []

    with torch.no_grad():
        for i,inputs in enumerate(input_sequences):
            signal_min = inputs.min()
            signal_max = inputs.max()
            inputs = (inputs - signal_min) / (signal_max - signal_min)

            plt.figure(figsize=(15, 10))
            plt.plot(inputs, label=f'输入序列 {i + 1}')

            inputs = torch.FloatTensor(inputs).unsqueeze(0).to(device)
            outputs = model(inputs)
            outputs =outputs*(tau_max-tau_min)+tau_min
            # 可视化输入序列
            plt.title(f'输入序列 {i + 1}: {taus[i]}，outputs:{outputs}')
            plt.legend()
            plt.tight_layout()
            #plt.savefig('input_sequences_visualization.png')
            plt.show()
            # 收集预测和真实值
            all_predictions.append(outputs.cpu().numpy())
        # 转换为numpy数组以便于计算统计值
        all_predictions_array = np.array(all_predictions)
        #all_predictions_array = [p * (tau_max - tau_min) + tau_min for p in all_predictions_array]
        # 计算统计数据
        mean_pred = np.mean(all_predictions_array)
        var_pred = np.var(all_predictions_array)
        min_pred = np.min(all_predictions_array)
        max_pred = np.max(all_predictions_array)

        # 展示统计信息
        print("\n预测值统计信息:")
        print(f"平均值: {mean_pred:.6f}")
        print(f"方差: {var_pred:.6f}")
        print(f"最小值: {min_pred:.6f}")
        print(f"最大值: {max_pred:.6f}")

        return all_predictions, {
            'mean': mean_pred,
            'variance': var_pred,
            'min': min_pred,
            'max': max_pred
        }

if __name__ == "__main__":
    model_path = "../phy_best_ringdown_model.pth"
    data_file = "stdtau0.002——10us.npz"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    predictions, stats = evaluate_model(model_path, data_file, device=device)
    print("评估完成，结果已保存")