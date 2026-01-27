# hapi_data_generator.py

from hapi import *
import numpy as np
import matplotlib.pyplot as plt


db_begin('hitran_data')
tablename='CO2'
fetch(tablename, 2, 1,9350, 9650)

data = LOCAL_TABLE_CACHE[tablename]['data']

# 只保留有代表性的强谱线（可选）
mask = data['sw'] > 1e-29
strong_lines = {}
for key in data:
    strong_lines[key] = np.array(data[key])[mask]
# strong_lines = data[data['sw'] > 1e-24]  # 自定义阈值

# 获取中心波数和线强列表
nu_list = strong_lines['nu']
S_list = strong_lines['sw']

def generate_sample_for_line(nu0, S_ref, T=296, P=1.0, x=1.0, L=1.0):
    nu_window = np.arange(nu0 - 0.5, nu0 + 0.5, 0.01)  # 100 点窗口
    try:
        nu, alpha = absorptionCoefficient_Voigt(
            SourceTables=tablename,
            OmegaGrid=nu_window,
            HITRAN_units=False,
            Environment={'T': T, 'p': P},
            Diluent={'self':x}
        )
        absorbance = alpha * P * L
        return absorbance, S_ref
    except:
        return None, None

# 6. 批量生成样本
X = []
Y = []

for nu0, S in zip(nu_list, S_list):
    x_i, y_i = generate_sample_for_line(nu0, S)
    if x_i is not None and len(x_i) == 100:
        X.append(x_i)
        Y.append(y_i)

X = np.array(X)
Y = np.array(Y).reshape(-1, 1)

# 7. 保存数据以供后续训练使用
np.save('X_absorbance_samples.npy', X)
np.save('Y_line_strengths.npy', Y)

# 8. 简单可视化一个样本
plt.plot(X[0])
plt.title(f'Sample Absorbance, Label S={Y[0][0]:.2e}')
plt.xlabel('Spectral Point')
plt.ylabel('Absorbance')
plt.grid(True)
plt.tight_layout()
plt.savefig('example_spectrum.png')
plt.show()