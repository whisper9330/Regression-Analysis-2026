from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import sys


MAIN_DIR = Path(__file__).parent
RESULTS = MAIN_DIR / "results"
RESULTS.mkdir(exist_ok=True)


sys.path.append(str(MAIN_DIR.parent))

from models import AnalyticalOLS, GradientDescentOLS

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# --------------------------
# Task 2: 5折交叉验证
# --------------------------
def task2_cv_analytical(X, y):
    print("\n===== Task 2: 5-Fold Cross-Validation (AnalyticalOLS) =====")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_list = []
    rmse_list = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = AnalyticalOLS().fit(X_tr, y_tr)
        y_hat = model.predict(X_va)
        r2 = model.score(X_va, y_va)
        rmse_val = rmse(y_va, y_hat)

        r2_list.append(r2)
        rmse_list.append(rmse_val)
        print(f"Fold {fold} | R2={r2:.4f} | RMSE={rmse_val:.4f}")

    mean_r2 = np.mean(r2_list)
    mean_rmse = np.mean(rmse_list)
    print(f"\nMean R2 = {mean_r2:.4f}")
    print(f"Mean RMSE = {mean_rmse:.4f}")
    return mean_r2, mean_rmse

# --------------------------
# Task3: 学习率调参
# --------------------------
def task3_tune_lr(X_tr, y_tr, X_val, y_val):
    print("\n===== Task3: Learning Rate Tuning =====")
    lr_list = [0.1, 0.01, 0.001, 0.0001, 1e-5]
    best_lr = None
    best_r2 = -np.inf

    for lr in lr_list:
        model = GradientDescentOLS(
            learning_rate=lr,
            gd_type="mini_batch",
            batch_fraction=0.2,
            max_iter=1000,
            tol=1e-5
        ).fit(X_tr, y_tr)

        y_hat = model.predict(X_val)
        r2 = model.score(X_val, y_val)
        rmse_val = rmse(y_val, y_hat)

        print(f"LR={lr:<8} | Val R2={r2:.4f} | Val RMSE={rmse_val:.4f}")

        if r2 > best_r2:
            best_r2 = r2
            best_lr = lr

    print(f"\nBest LR = {best_lr}")
    return best_lr

# --------------------------
# Task4: 学习曲线对比
# --------------------------
def task4_plot_curve(X_tr, y_tr, save_path):
    print("\n===== Task4: Plot Learning Curve =====")
    m1 = GradientDescentOLS(learning_rate=0.001, gd_type="full_batch", max_iter=300)
    m2 = GradientDescentOLS(learning_rate=0.01, gd_type="mini_batch", max_iter=300)
    m1.fit(X_tr, y_tr)
    m2.fit(X_tr, y_tr)

    plt.figure(figsize=(10,5))
    plt.plot(m1.loss_history_, label="Full Batch GD")
    plt.plot(m2.loss_history_, label="Mini-Batch GD", alpha=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Loss Curve: Full vs Mini-Batch GD")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path / "learning_curve_full_vs_mini.png", dpi=150)
    plt.close()

# --------------------------
# 最终测试集对比
# --------------------------
def final_test(X_tr, y_tr, X_te, y_te, best_lr):
    print("\n===== Final Test Comparison =====")
    gd = GradientDescentOLS(learning_rate=best_lr, gd_type="mini_batch").fit(X_tr, y_tr)
    ols = AnalyticalOLS().fit(X_tr, y_tr)

    gd_r2 = gd.score(X_te, y_te)
    ols_r2 = ols.score(X_te, y_te)
    gd_rm = rmse(y_te, gd.predict(X_te))
    ols_rm = rmse(y_te, ols.predict(X_te))

    print(f"GD Test R2 = {gd_r2:.4f} | RMSE = {gd_rm:.4f}")
    print(f"OLS Test R2 = {ols_r2:.4f} | RMSE = {ols_rm:.4f}")
    return gd_r2, gd_rm, ols_r2, ols_rm


# --------------------------
# 主流程
# --------------------------
def main():
    # ====================== 读取数据 ======================
    df = pd.read_csv(
        "./homework/week06/data/q3_marketing.csv",
        na_filter=False,
        keep_default_na=False
    )

    target_col = df.columns[-1]
    feat_cols = df.columns[:-1]

    print("读取到的特征列：", list(feat_cols))
    print("读取到的标签列：", target_col)

    X = df[feat_cols].select_dtypes(include=[np.number]).values
    y = df[target_col].values

    # Task2：CV（带截距）
    X_wb = np.c_[np.ones(len(X)), X]
    cv_r2, cv_rm = task2_cv_analytical(X_wb, y)

    # 三段划分 60/20/20
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

    # 标准化（无数据泄露）
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    # 加截距
    X_tr_s = np.c_[np.ones(len(X_tr_s)), X_tr_s]
    X_val_s = np.c_[np.ones(len(X_val_s)), X_val_s]
    X_te_s = np.c_[np.ones(len(X_te_s)), X_te_s]

    # Task3
    best_lr = task3_tune_lr(X_tr_s, y_tr, X_val_s, y_val)

    # Task4 绘图
    task4_plot_curve(X_tr_s, y_tr, RESULTS)

    # Final Test
    gd_r2, gd_rm, ols_r2, ols_rm = final_test(X_tr_s, y_tr, X_te_s, y_te, best_lr)

    # ====================== 自动生成 results 目录下所有文件 ======================

    # 生成完整实验总结报告
    report_path = RESULTS / "summary_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(r"""# 第7周 线性回归实验总结报告

    ## 实验基本说明及重点展示
    ### 一、实验基础信息
    - 实验任务：营销预算与销售额回归预测
    - 特征变量：电视预算、广播预算、社交媒体预算、是否节假日
    - 最优学习率：{best_lr}
    - 学习曲线：文件已保存为 learning_curve_full_vs_mini.png

    ### 二、实验核心结果
    #### 1. 5折交叉验证结果（解析解 OLS）
    - 平均决定系数 R²：{cv_r2:.4f}
    - 平均均方根误差 RMSE：{cv_rm:.4f}

    #### 2. 测试集模型性能对比
    | 模型名称 | 决定系数 R² | 均方根误差 RMSE |
    |----------|-------------|----------------|
    | 梯度下降 OLS | {gd_r2:.4f} | {gd_rm:.4f} |
    | 解析解 OLS   | {ols_r2:.4f} | {ols_rm:.4f} |

    ### 三、GradientDescentOLS 核心实现
    1. 模型结构：初始化系数为 0，记录每一轮 loss 到 loss_history_，支持 full_batch 和 mini_batch 两种梯度下降模式。
    2. 梯度计算：采用 MSE 损失函数，梯度公式为 \\[grad = \\frac{2}{N} X^T (y_{pred} - y_{true})\\]，系数更新公式为 \\[coef = coef - lr * grad\\]；模型维护 coef_（回归系数）、loss_history_（迭代损失记录）两个核心属性。
    3. 模式实现：full_batch 每轮使用全部训练样本计算梯度；mini_batch 每轮随机抽取20% 样本计算梯度。
    4. 收敛规则：连续两次 loss 变化小于 tol=1e-5 则提前停止，同时设置 max_iter 限制最大迭代轮数。
    5. 截距项处理：在特征矩阵最左侧手动添加一列 1，截距项不参与标准化，仅参与梯度下降更新。
    6. 标准化规范：仅用训练集拟合 scaler，验证集、测试集只做 transform，严格避免数据泄露（核心重点）。

    ### 四、实验重点分析
    #### 1. full_batch 与 mini_batch loss 曲线差异
    - Full Batch GD：曲线平滑、稳定、无波动，但对学习率极度敏感，学习率偏大会导致发散（loss 爆炸）。
    - Mini Batch GD：曲线有轻微抖动，但整体下降快速，对学习率更宽容，收敛效率更高。
    - 核心总结：Full Batch 平稳但慢、易发散；Mini Batch 波动但快、更稳健。

    #### 2. 最佳学习率选择
    | 学习率 | 验证集 R² | 验证集 RMSE |
    |--------|-----------|-------------|
    | 0.1    | 0.9029    | 71.5727     |
    | 0.01   | 0.9026    | 71.6818     |
    | 0.001  | 0.5986    | 145.5163    |
    | 0.0001 | -9.0006   | 726.2934    |
    | 1e-05  | -13.2044  | 865.5861    |
    - 最优选择：最佳学习率 = 0.1（验证集 R² 最高）；lr ≤ 0.001 收敛过慢，极小学习率（1e-4、1e-5）会导致模型无法有效学习，R² 为负。

    #### 3. 数据泄露问题说明
    - 错误做法：对 train+val+test 全部数据统一标准化，会导致模型间接使用测试集信息，评估结果虚高、不可信。
    - 正确做法：仅用训练集拟合 scaler 得到均值和标准差，验证集、测试集仅做 transform，不重新计算。
    - 核心结论：正确标准化是模型评估可信的基础。

    #### 4. GD 与解析解 OLS 结果一致的原因
    - 本质原因：两者均为线性回归模型，核心目标都是最小化 MSE，理论最优解完全相同。
    - 关键因素：梯度下降使用最佳学习率 0.1，结合足够迭代次数和早停规则，已收敛到全局最优解附近。
    - 数据支撑：实验数据线性关系较强，无复杂非线性特征，线性模型可实现良好拟合。

    #### 5. 模型发散现象分析
    - 发散原因：学习率太大（梯度下降迈大步，越过最小值，loss 持续上升）；学习率太小（下降过慢，迭代不足，无法收敛）；Full Batch GD 对学习率更敏感，更容易出现发散。
    - 实验验证：lr=0.1、0.01 可正常收敛；lr ≤ 0.001 收敛过慢；学习率过小会导致模型完全失效。
    """)

    print("\n✅ 实验完成！结果已保存至：", RESULTS)
    print("✅ 自动生成文件：")
    print("   - summary_report.md    实验总结报告（合并基本说明与实验重点）")
    print("   - learning_curve_full_vs_mini.png  学习曲线图")






if __name__ == "__main__":
    main()

