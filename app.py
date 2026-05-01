import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import japanize_matplotlib

# ページ全体のレイアウト設定
st.set_page_config(page_title="軍備拡張の数理モデル", layout="wide")

# ==========================================
# 1. ルンゲ・クッタ法の計算関数
# ==========================================
def solve_richardson(a1, b1, a2, b2, c1=5.0, c2=5.0, x0=40.0, y0=40.0, dt=0.1, t_max=75.0):
    t_list = np.arange(0, t_max + dt * 0.1, dt)
    n_steps = len(t_list)
    x_list = np.zeros(n_steps)
    y_list = np.zeros(n_steps)
    x_list[0], y_list[0] = x0, y0
    
    for i in range(n_steps - 1):
        x_n, y_n = x_list[i], y_list[i]
        
        kx1 = a1 * y_n - b1 * x_n + c1
        ky1 = a2 * x_n - b2 * y_n + c2
        
        kx2 = a1 * (y_n + ky1 * dt / 2) - b1 * (x_n + kx1 * dt / 2) + c1
        ky2 = a2 * (x_n + kx1 * dt / 2) - b2 * (y_n + ky1 * dt / 2) + c2
        
        kx3 = a1 * (y_n + ky2 * dt / 2) - b1 * (x_n + kx2 * dt / 2) + c1
        ky3 = a2 * (x_n + kx2 * dt / 2) - b2 * (y_n + ky2 * dt / 2) + c2
        
        kx4 = a1 * (y_n + ky3 * dt) - b1 * (x_n + kx3 * dt) + c1
        ky4 = a2 * (x_n + kx3 * dt) - b2 * (y_n + ky3 * dt) + c2
        
        x_list[i+1] = x_n + (kx1 + 2*kx2 + 2*kx3 + kx4) * dt / 6
        y_list[i+1] = y_n + (ky1 + 2*ky2 + 2*ky3 + ky4) * dt / 6
        
    return t_list, x_list, y_list

# ==========================================
# 2. サイドバー（入力部分）の作成
# ==========================================
st.sidebar.markdown("### シミュレーション設定")
t_max = st.sidebar.slider("シミュレーション期間", min_value=5.0, max_value=200.0, value=75.0, step=1.0)

st.sidebar.divider()

st.sidebar.markdown("#### A国のパラメータ")
a1 = st.sidebar.slider("対抗係数 (a1)", 0.0, 1.0, 0.20, 0.01)
b1 = st.sidebar.slider("抑制係数 (b1)", 0.0, 1.0, 0.30, 0.01)
c1 = st.sidebar.slider("不満項 (c1)", 0.0, 10.0, 5.0, 0.1)

st.sidebar.divider()

st.sidebar.markdown("#### B国のパラメータ")
a2 = st.sidebar.slider("対抗係数 (a2)", 0.0, 1.0, 0.40, 0.01)
b2 = st.sidebar.slider("抑制係数 (b2)", 0.0, 1.0, 0.40, 0.01)
c2 = st.sidebar.slider("不満項 (c2)", 0.0, 10.0, 5.0, 0.1)

# ==========================================
# 3. シミュレーション実行と状態・平衡点の計算
# ==========================================
t_val, x_val, y_val = solve_richardson(a1, b1, a2, b2, c1, c2, t_max=t_max)
final_x, final_y = x_val[-1], y_val[-1]

# 安定性の判定
det = b1 * b2 - a1 * a2
is_stable = det > 0

# 理論上の真の平衡点を連立方程式から計算
eq_x, eq_y = None, None
if is_stable:
    eq_x = (c1 * b2 + c2 * a1) / det
    eq_y = (b1 * c2 + a2 * c1) / det

# ==========================================
# 4. メイン画面：ステータス表示
# ==========================================
st.markdown("## 軍備拡張の数理モデル (Richardson Model)")

st.markdown(
    """
    <style>
    [data-testid="stMetric"] { display: flex; flex-direction: column; align-items: center; text-align: center; }
    [data-testid="stMetricLabel"] { display: flex; justify-content: center; width: 100%; }
    [data-testid="stMetricValue"] { display: flex; justify-content: center; width: 100%; }
    </style>
    """,
    unsafe_allow_html=True
)

def format_large_val(val):
    return f"{val:.2e}" if abs(val) >= 100000 else f"{val:.1f}"

met1, met2, met3, met4 = st.columns(4)
met1.metric("経過時間", f"{t_max:.1f}")
met2.metric("A国の軍備量 (X)", format_large_val(final_x))
met3.metric("B国の軍備量 (Y)", format_large_val(final_y))

if is_stable:
    met4.success("収束（安定）")
    st.info("⚖️ 平衡状態に向かっています。 ($b_1 b_2 - a_1 a_2 > 0$)")
elif det == 0:
    met4.warning("均衡境界（臨界）")
    st.warning("⚠️ 安定と暴走の境界線上にあります。 ($b_1 b_2 - a_1 a_2 = 0$)")
else:
    met4.error("発散（暴走）")
    st.error("💥 制御不能な軍拡競争が発生しています。 ($b_1 b_2 - a_1 a_2 < 0$)")

# ==========================================
# 5. メイン画面：棒グラフの描画
# ==========================================
fig_bar, ax_bar = plt.subplots(figsize=(8, 1.2))
ax_bar.bar(["A国", "B国"], [final_x, final_y], color=['#4169E1', '#CD5C5C'], width=0.4)
ax_bar.set_ylim(0, max(100, max(final_x, final_y) * 1.2))
ax_bar.set_ylabel("軍備量")
st.pyplot(fig_bar)

# ==========================================
# 6 & 7. メイン画面：グラフを1つのキャンバスに統合
# ==========================================
max_val_all = max(100, max(max(x_val), max(y_val)) * 1.1)
if is_stable:
    max_val_all = max(max_val_all, max(eq_x, eq_y) * 1.2)

fig_main, (ax_line, ax_phase) = plt.subplots(1, 2, figsize=(12, 5))

# 左側：時間発展グラフ
ax_line.set_title("軍備拡張の時間発展", fontsize=14, pad=10)
ax_line.plot(t_val, x_val, label='A国 (X)', color='#4169E1', linewidth=2)
ax_line.plot(t_val, y_val, label='B国 (Y)', color='#CD5C5C', linewidth=2)

# 平衡点の水平線を点線で追加
if is_stable:
    ax_line.axhline(y=eq_x, color='#4169E1', linestyle=':', alpha=0.5)
    ax_line.axhline(y=eq_y, color='#CD5C5C', linestyle=':', alpha=0.5)
ax_line.set_xlabel("時間 ($t$)")
ax_line.set_ylabel("軍備量")
ax_line.set_xlim(0, max(t_max, 50)) # 最低でも横軸を50は確保
ax_line.set_ylim(0, max_val_all)
ax_line.legend(loc='upper left')
ax_line.grid(True, linestyle='--', alpha=0.7)

# 右側：相図
ax_phase.set_title("相図", fontsize=14, pad=10)
ax_phase.plot(x_val, y_val, color='#9b59b6', linewidth=2.5, label='軌跡')
ax_phase.scatter([x_val[0]], [y_val[0]], color='#2ecc71', s=100, zorder=5, label='初期状態')
ax_phase.scatter([final_x], [final_y], color='#3498db', s=100, zorder=5, label='最終状態(t)')

# 理論上の真の平衡点を赤い星マークでプロット
if is_stable:
    ax_phase.scatter([eq_x], [eq_y], color='#e74c3c', marker='*', s=300, zorder=6, label='真の平衡点(∞)')

ax_phase.set_xlabel("A国の軍備量 (X)")
ax_phase.set_ylabel("B国の軍備量 (Y)")
ax_phase.set_xlim(0, max_val_all)
ax_phase.set_ylim(0, max_val_all)
ax_phase.legend(loc='upper left')
ax_phase.grid(True, linestyle='--', alpha=0.7)

fig_main.subplots_adjust(wspace=0.3)
st.pyplot(fig_main)