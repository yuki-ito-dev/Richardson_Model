import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import japanize_matplotlib
from mpl_toolkits.mplot3d import Axes3D

# ページ全体のレイアウト設定
st.set_page_config(page_title="軍備拡張の数理モデル", layout="wide")

# ==========================================
# 1. ルンゲ・クッタ法の計算関数
# ==========================================
def solve_richardson(a1, b1, a2, b2, c1, c2, L1, L2, x0=40.0, y0=40.0, dt=0.1, t_max=75.0):
    t_list = np.arange(0, t_max + dt * 0.1, dt)
    n_steps = len(t_list)
    x_list = np.zeros(n_steps)
    y_list = np.zeros(n_steps)
    x_list[0], y_list[0] = x0, y0
    
    for i in range(n_steps - 1):
        x_n, y_n = x_list[i], y_list[i]
        
        def dx_dt(x, y): return a1 * y - b1 * x - L1 * (x**2) + c1
        def dy_dt(x, y): return a2 * x - b2 * y - L2 * (y**2) + c2
        
        kx1 = dx_dt(x_n, y_n)
        ky1 = dy_dt(x_n, y_n)
        
        kx2 = dx_dt(x_n + kx1 * dt / 2, y_n + ky1 * dt / 2)
        ky2 = dy_dt(x_n + kx1 * dt / 2, y_n + ky1 * dt / 2)
        
        kx3 = dx_dt(x_n + kx2 * dt / 2, y_n + ky2 * dt / 2)
        ky3 = dy_dt(x_n + kx2 * dt / 2, y_n + ky2 * dt / 2)
        
        kx4 = dx_dt(x_n + kx3 * dt, y_n + ky3 * dt)
        ky4 = dy_dt(x_n + kx3 * dt, y_n + ky3 * dt)
        
        x_list[i+1] = x_n + (kx1 + 2*kx2 + 2*kx3 + kx4) * dt / 6
        y_list[i+1] = y_n + (ky1 + 2*ky2 + 2*ky3 + ky4) * dt / 6
        
    return t_list, x_list, y_list

def solve_richardson_3(a12, a13, b1, c1, a21, a23, b2, c2, a31, a32, b3, c3, L1=0.0, L2=0.0, L3=0.0, x0=40.0, y0=40.0, z0=40.0, dt=0.1, t_max=75.0):
    t_list = np.arange(0, t_max + dt * 0.1, dt)
    n_steps = len(t_list)
    x_list = np.zeros(n_steps)
    y_list = np.zeros(n_steps)
    z_list = np.zeros(n_steps)
    x_list[0], y_list[0], z_list[0] = x0, y0, z0
    
    for i in range(n_steps - 1):
        x_n, y_n, z_n = x_list[i], y_list[i], z_list[i]
        
        def dx_dt(x, y, z): return a12 * y + a13 * z - b1 * x - L1 * x * abs(x) + c1
        def dy_dt(x, y, z): return a21 * x + a23 * z - b2 * y - L2 * y * abs(y) + c2
        def dz_dt(x, y, z): return a31 * x + a32 * y - b3 * z - L3 * z * abs(z) + c3
        
        kx1, ky1, kz1 = dx_dt(x_n, y_n, z_n), dy_dt(x_n, y_n, z_n), dz_dt(x_n, y_n, z_n)
        kx2, ky2, kz2 = dx_dt(x_n + kx1*dt/2, y_n + ky1*dt/2, z_n + kz1*dt/2), dy_dt(x_n + kx1*dt/2, y_n + ky1*dt/2, z_n + kz1*dt/2), dz_dt(x_n + kx1*dt/2, y_n + ky1*dt/2, z_n + kz1*dt/2)
        kx3, ky3, kz3 = dx_dt(x_n + kx2*dt/2, y_n + ky2*dt/2, z_n + kz2*dt/2), dy_dt(x_n + kx2*dt/2, y_n + ky2*dt/2, z_n + kz2*dt/2), dz_dt(x_n + kx2*dt/2, y_n + ky2*dt/2, z_n + kz2*dt/2)
        kx4, ky4, kz4 = dx_dt(x_n + kx3*dt, y_n + ky3*dt, z_n + kz3*dt), dy_dt(x_n + kx3*dt, y_n + ky3*dt, z_n + kz3*dt), dz_dt(x_n + kx3*dt, y_n + ky3*dt, z_n + kz3*dt)
        
        x_list[i+1] = x_n + (kx1 + 2*kx2 + 2*kx3 + kx4) * dt / 6
        y_list[i+1] = y_n + (ky1 + 2*ky2 + 2*ky3 + ky4) * dt / 6
        z_list[i+1] = z_n + (kz1 + 2*kz2 + 2*kz3 + kz4) * dt / 6
        
    return t_list, x_list, y_list, z_list

# 【追加】Nカ国向けの汎用計算エンジン (NumPy行列演算)
def solve_richardson_N(A, C, L, x0, dt=0.1, t_max=75.0):
    t_list = np.arange(0, t_max + dt * 0.1, dt)
    n_steps = len(t_list)
    N = len(C)
    X = np.zeros((n_steps, N))
    X[0] = x0
    
    for i in range(n_steps - 1):
        x_n = X[i]
        def dX_dt(x): return A @ x + C - L * x * np.abs(x)
        k1 = dX_dt(x_n)
        k2 = dX_dt(x_n + k1 * dt / 2)
        k3 = dX_dt(x_n + k2 * dt / 2)
        k4 = dX_dt(x_n + k3 * dt)
        X[i+1] = x_n + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
        
    return t_list, X

# ==========================================
# 2. サイドバー (入力部分)の作成
# ==========================================
st.sidebar.markdown("### シミュレーション設定")

preset = st.sidebar.selectbox("歴史的シナリオ・プリセット", [
    "カスタム設定", 
    "冷戦型 (2カ国間の激しい敵対)", 
    "安定的な平和 (2カ国間の相互抑制)", 
    "核の傘と同盟 (3カ国間のフリーライダー問題)",
    "三つ巴のジレンマ (3カ国間の相互敵対)"
])

defaults = {
    "model": "2カ国 線形モデル (基本)", "t": 75.0,
    "a1": 0.20, "b1": 0.30, "c1": 5.0, "L1": 0.005,
    "a2": 0.40, "b2": 0.40, "c2": 5.0, "L2": 0.005,
    "a12": 0.20, "a13": 0.20, "b1_3": 0.30, "c1_3": 5.0, "L1_3": 0.005,
    "a21": 0.40, "a23": 0.40, "b2_3": 0.40, "c2_3": 5.0, "L2_3": 0.005,
    "a31": 0.30, "a32": 0.30, "b3_3": 0.30, "c3_3": 5.0, "L3_3": 0.005
}

if preset == "冷戦型 (2カ国間の激しい敵対)":
    defaults.update({"a1": 0.80, "b1": 0.20, "a2": 0.80, "b2": 0.20, "t": 150.0})
elif preset == "安定的な平和 (2カ国間の相互抑制)":
    defaults.update({"a1": 0.10, "b1": 0.60, "a2": 0.10, "b2": 0.60})
elif preset == "核の傘と同盟 (3カ国間のフリーライダー問題)":
    defaults.update({"model": "3カ国 線形モデル (同盟と敵対)", "a12": -0.80, "a13": 0.50, "a21": -0.80, "a23": 0.50, "b1_3": 0.20, "b2_3": 0.40})
elif preset == "三つ巴のジレンマ (3カ国間の相互敵対)":
    defaults.update({"model": "3カ国 線形モデル (同盟と敵対)", "a12": 0.60, "a13": 0.60, "a21": 0.60, "a23": 0.60, "a31": 0.60, "a32": 0.60})

# 【追加】Nカ国モデルの選択肢を追加
model_options = [
    "2カ国 線形モデル (基本)", 
    "2カ国 非線形モデル (抑制項の導入)", 
    "3カ国 線形モデル (同盟と敵対)", 
    "3カ国 非線形モデル (抑制項の導入)",
    "Nカ国 線形モデル (一般化行列)",
    "Nカ国 非線形モデル (一般化行列)"
]

model_type = st.sidebar.radio("モデル選択", model_options, index=model_options.index(defaults["model"]))

t_max = st.sidebar.slider("シミュレーション期間", min_value=5.0, max_value=300.0, value=defaults["t"], step=1.0)

st.sidebar.divider()

if model_type in ["2カ国 線形モデル (基本)", "2カ国 非線形モデル (抑制項の導入)"]:
    with st.sidebar.expander("A国のパラメータ", expanded=True):
        a1 = st.slider("対抗係数 (a1)", 0.0, 1.0, defaults["a1"], 0.01)
        b1 = st.slider("抑制係数 (b1)", 0.0, 1.0, defaults["b1"], 0.01)
        c1 = st.slider("不満項 (c1)", 0.0, 10.0, defaults["c1"], 0.1)
        L1 = 0.0
        if model_type == "2カ国 非線形モデル (抑制項の導入)":
            L1 = st.slider("非線形抑制係数 (L1) ※限界", 0.000, 0.050, defaults["L1"], 0.001, key="L1_2")

    with st.sidebar.expander("B国のパラメータ", expanded=True):
        a2 = st.slider("対抗係数 (a2)", 0.0, 1.0, defaults["a2"], 0.01)
        b2 = st.slider("抑制係数 (b2)", 0.0, 1.0, defaults["b2"], 0.01)
        c2 = st.slider("不満項 (c2)", 0.0, 10.0, defaults["c2"], 0.1)
        L2 = 0.0
        if model_type == "2カ国 非線形モデル (抑制項の導入)":
            L2 = st.slider("非線形抑制係数 (L2) ※限界", 0.000, 0.050, defaults["L2"], 0.001, key="L2_2")

elif model_type in ["3カ国 線形モデル (同盟と敵対)", "3カ国 非線形モデル (抑制項の導入)"]:
    with st.sidebar.expander("A国のパラメータ", expanded=True):
        a12 = st.slider("対B国 対抗係数 (a12)", -1.0, 1.0, defaults["a12"], 0.01)
        a13 = st.slider("対C国 対抗係数 (a13)", -1.0, 1.0, defaults["a13"], 0.01)
        b1_3 = st.slider("A国 抑制係数 (b1)", 0.0, 1.0, defaults["b1_3"], 0.01)
        c1_3 = st.slider("A国 不満項 (c1)", 0.0, 10.0, defaults["c1_3"], 0.1)
        L1_3 = 0.0
        if model_type == "3カ国 非線形モデル (抑制項の導入)":
            L1_3 = st.slider("非線形抑制係数 (L1)", 0.000, 0.050, defaults["L1_3"], 0.001, key="L1_3")

    with st.sidebar.expander("B国のパラメータ", expanded=True):
        a21 = st.slider("対A国 対抗係数 (a21)", -1.0, 1.0, defaults["a21"], 0.01)
        a23 = st.slider("対C国 対抗係数 (a23)", -1.0, 1.0, defaults["a23"], 0.01)
        b2_3 = st.slider("B国 抑制係数 (b2)", 0.0, 1.0, defaults["b2_3"], 0.01)
        c2_3 = st.slider("B国 不満項 (c2)", 0.0, 10.0, defaults["c2_3"], 0.1)
        L2_3 = 0.0
        if model_type == "3カ国 非線形モデル (抑制項の導入)":
            L2_3 = st.slider("非線形抑制係数 (L2)", 0.000, 0.050, defaults["L2_3"], 0.001, key="L2_3")

    with st.sidebar.expander("C国のパラメータ", expanded=True):
        a31 = st.slider("対A国 対抗係数 (a31)", -1.0, 1.0, defaults["a31"], 0.01)
        a32 = st.slider("対B国 対抗係数 (a32)", -1.0, 1.0, defaults["a32"], 0.01)
        b3_3 = st.slider("C国 抑制係数 (b3)", 0.0, 1.0, defaults["b3_3"], 0.01)
        c3_3 = st.slider("C国 不満項 (c3)", 0.0, 10.0, defaults["c3_3"], 0.1)
        L3_3 = 0.0
        if model_type == "3カ国 非線形モデル (抑制項の導入)":
            L3_3 = st.slider("非線形抑制係数 (L3)", 0.000, 0.050, defaults["L3_3"], 0.001, key="L3_3")

# 【変更】Nカ国用のサイドバーUI (対抗係数→抑制係数→不満項の順に修正)
else:
    N = st.sidebar.number_input("国の数 (N)", min_value=2, max_value=10, value=4, step=1)
    
    A_matrix = np.zeros((N, N))
    C_vector = np.zeros(N)
    L_vector = np.zeros(N)
    
    for i in range(N):
        with st.sidebar.expander(f"国{i+1}のパラメータ", expanded=True):
            # まず対抗係数 (i != j) を配置
            for j in range(N):
                if i != j:
                    A_matrix[i, j] = st.slider(f"対国{j+1} 対抗係数 (A{i+1}{j+1})", -1.0, 1.0, 0.20, 0.01, key=f"A_{i}_{j}")
            
            # 次に抑制係数 (i == j) を不満項のすぐ上に配置
            A_matrix[i, i] = st.slider(f"国{i+1} 抑制係数 (A{i+1}{i+1}) ※マイナスのみ", -1.0, 0.0, -0.30, 0.01, key=f"A_{i}_{i}")
            
            # 不満項
            C_vector[i] = st.slider(f"国{i+1} 不満項 (c{i+1})", 0.0, 10.0, 5.0, 0.1, key=f"C_{i}")
            
            # 非線形抑制係数
            if "非線形" in model_type:
                L_vector[i] = st.slider(f"非線形抑制係数 (L{i+1})", 0.000, 0.050, 0.005, 0.001, key=f"L_{i}")

# ==========================================
# 3. シミュレーション実行と状態・平衡点の計算
# ==========================================
if model_type in ["2カ国 線形モデル (基本)", "2カ国 非線形モデル (抑制項の導入)"]:
    t_val, x_val, y_val = solve_richardson(a1, b1, a2, b2, c1, c2, L1, L2, t_max=t_max)
    final_x, final_y = x_val[-1], y_val[-1]

    A_matrix = np.array([
        [-b1,  a1],
        [ a2, -b2]
    ])
    eigenvalues = np.linalg.eigvals(A_matrix)

    det = b1 * b2 - a1 * a2
    is_linear_stable = det > 0

    eq_x, eq_y = None, None
    if model_type == "2カ国 線形モデル (基本)" and is_linear_stable:
        eq_x = (c1 * b2 + c2 * a1) / det
        eq_y = (b1 * c2 + a2 * c1) / det

elif model_type in ["3カ国 線形モデル (同盟と敵対)", "3カ国 非線形モデル (抑制項の導入)"]:
    t_val, x_val, y_val, z_val = solve_richardson_3(a12, a13, b1_3, c1_3, a21, a23, b2_3, c2_3, a31, a32, b3_3, c3_3, L1_3, L2_3, L3_3, t_max=t_max)
    final_x, final_y, final_z = x_val[-1], y_val[-1], z_val[-1]
    
    A_matrix = np.array([
        [-b1_3,  a12,  a13],
        [ a21, -b2_3,  a23],
        [ a31,  a32, -b3_3]
    ])
    C_vector = np.array([-c1_3, -c2_3, -c3_3])
    eigenvalues = np.linalg.eigvals(A_matrix)
    
    if model_type == "3カ国 線形モデル (同盟と敵対)":
        try:
            is_linear_stable = np.all(np.real(eigenvalues) < 0)
            if is_linear_stable:
                eq_3d = np.linalg.solve(A_matrix, C_vector)
                eq_x, eq_y, eq_z = eq_3d[0], eq_3d[1], eq_3d[2]
            else:
                eq_x, eq_y, eq_z = None, None, None
        except np.linalg.LinAlgError:
            is_linear_stable = False
            eq_x, eq_y, eq_z = None, None, None
    else:
        is_linear_stable = False
        eq_x, eq_y, eq_z = None, None, None

# 【追加】Nカ国の計算処理
else:
    x0 = np.full(N, 40.0)
    t_val, X_val = solve_richardson_N(A_matrix, C_vector, L_vector, x0, t_max=t_max)
    final_X = X_val[-1]
    
    eigenvalues = np.linalg.eigvals(A_matrix)
    is_linear_stable = np.all(np.real(eigenvalues) < 0)
    
    if "線形" in model_type:
        try:
            if is_linear_stable:
                eq_X = np.linalg.solve(A_matrix, -C_vector)
            else:
                eq_X = None
        except np.linalg.LinAlgError:
            is_linear_stable = False
            eq_X = None
    else:
        is_linear_stable = False
        eq_X = None

# ==========================================
# 4. メイン画面：ステータス表示
# ==========================================
st.markdown("## 軍備拡張の数理モデル (Richardson Model)")

with st.expander("\CID{1884} 現在のモデルの方程式"):
    if model_type == "2カ国 線形モデル (基本)":
        st.markdown(r"""
        $$
        \begin{cases}
        \frac{dx}{dt} = a_1 y - b_1 x + c_1 \\
        \frac{dy}{dt} = a_2 x - b_2 y + c_2
        \end{cases}
        $$
        """)
    elif model_type == "2カ国 非線形モデル (抑制項の導入)":
        st.markdown(r"""
        $$
        \begin{cases}
        \frac{dx}{dt} = a_1 y - b_1 x - L_1 x^2 + c_1 \\
        \frac{dy}{dt} = a_2 x - b_2 y - L_2 y^2 + c_2
        \end{cases}
        $$
        """)
    elif model_type == "3カ国 線形モデル (同盟と敵対)":
        st.markdown(r"""
        $$
        \begin{cases}
        \frac{dx}{dt} = a_{12} y + a_{13} z - b_1 x + c_1 \\
        \frac{dy}{dt} = a_{21} x + a_{23} z - b_2 y + c_2 \\
        \frac{dz}{dt} = a_{31} x + a_{32} y - b_3 z + c_3
        \end{cases}
        $$
        """)
    elif model_type == "3カ国 非線形モデル (抑制項の導入)":
        st.markdown(r"""
        $$
        \begin{cases}
        \frac{dx}{dt} = a_{12} y + a_{13} z - b_1 x - L_1 x |x| + c_1 \\
        \frac{dy}{dt} = a_{21} x + a_{23} z - b_2 y - L_2 y |y| + c_2 \\
        \frac{dz}{dt} = a_{31} x + a_{32} y - b_3 z - L_3 z |z| + c_3
        \end{cases}
        $$
        """)
    elif model_type == "Nカ国 線形モデル (一般化行列)":
        st.markdown(r"""
        $$
        \frac{d\mathbf{x}}{dt} = A\mathbf{x} + \mathbf{c}
        $$
        """)
    elif model_type == "Nカ国 非線形モデル (一般化行列)":
        st.markdown(r"""
        $$
        \frac{d\mathbf{x}}{dt} = A\mathbf{x} + \mathbf{c} - L \odot \mathbf{x} \odot |\mathbf{x}|
        $$
        """)
    
    st.markdown("---")
    
    # 【修正】文言のアップデートと動的なLaTeX行列の作成関数
    if model_type in ["2カ国 線形モデル (基本)", "2カ国 非線形モデル (抑制項の導入)"]:
        st.markdown("**\CID{1819} リアルタイム行列表現（2カ国モデル）**")
        st.markdown(r"システムは $\frac{d\mathbf{x}}{dt} = A\mathbf{x} + \mathbf{c} + \text{非線形項}$ のように行列で定式化されます。2カ国モデルでは、この行列表現を用いて相互作用を計算しています。")
    elif model_type in ["3カ国 線形モデル (同盟と敵対)", "3カ国 非線形モデル (抑制項の導入)"]:
        st.markdown("**\CID{1819} リアルタイム行列表現（3カ国モデル）**")
        st.markdown(r"システムは $\frac{d\mathbf{x}}{dt} = A\mathbf{x} + \mathbf{c} + \text{非線形項}$ のように行列で定式化されます。3カ国モデルでは、この行列表現を用いて相互作用を計算しています。")
    else:
        st.markdown("**\CID{1819} リアルタイム行列表現（一般化Nカ国モデル）**")
        st.markdown(r"システムは $\frac{d\mathbf{x}}{dt} = A\mathbf{x} + \mathbf{c} + \text{非線形項}$ のように行列で定式化されます。Nカ国モデルでは、この行列表現を用いて任意の多国間相互作用を計算しています。")

    def matrix_to_latex(mat):
        lines = [" & ".join([f"{val:.2f}" for val in row]) for row in mat]
        return "\\begin{pmatrix} " + " \\\\ ".join(lines) + " \\end{pmatrix}"

    if model_type in ["2カ国 線形モデル (基本)", "2カ国 非線形モデル (抑制項の導入)"]:
        st.markdown(fr"""
        $$
        A = \begin{{pmatrix}} {-b1:.2f} & {a1:.2f} \\ {a2:.2f} & {-b2:.2f} \end{{pmatrix}}, \quad
        \mathbf{{c}} = \begin{{pmatrix}} {c1:.2f} \\ {c2:.2f} \end{{pmatrix}}
        $$
        """)
    elif model_type in ["3カ国 線形モデル (同盟と敵対)", "3カ国 非線形モデル (抑制項の導入)"]:
        st.markdown(fr"""
        $$
        A = \begin{{pmatrix}} {-b1_3:.2f} & {a12:.2f} & {a13:.2f} \\ {a21:.2f} & {-b2_3:.2f} & {a23:.2f} \\ {a31:.2f} & {a32:.2f} & {-b3_3:.2f} \end{{pmatrix}}, \quad
        \mathbf{{c}} = \begin{{pmatrix}} {c1_3:.2f} \\ {c2_3:.2f} \\ {c3_3:.2f} \end{{pmatrix}}
        $$
        """)
    else:
        A_str = matrix_to_latex(A_matrix)
        C_str = matrix_to_latex(C_vector.reshape(-1, 1))
        st.markdown(fr"""
        $$
        A = {A_str}, \quad
        \mathbf{{c}} = {C_str}
        $$
        """)

with st.expander("\CID{1865} 数学的解析 (固有値とシステムの安定性)", expanded=True):
    ev_str = ", ".join([f"{ev.real:.3f} + {ev.imag:.3f}i" if ev.imag != 0 else f"{ev.real:.3f}" for ev in eigenvalues])
    st.write(f"**系行列の固有値:** `{ev_str}`")
    
    reals = np.real(eigenvalues)
    imags = np.imag(eigenvalues)
    
    if np.all(reals < 0):
        if np.any(np.abs(imags) > 1e-5):
            st.success("\UTF{2705} **安定な渦状収束 (Spiral Sink):** 全ての固有値の実部が負で、虚部が存在します。軍備は振動しながら均衡点へ収束します。")
        else:
            st.success("\UTF{2705} **安定な結節点 (Stable Node):** 全ての固有値が負の実数です。軍備は振動することなく滑らかに均衡点へ収束します。")
    elif np.any(reals > 0):
        st.error("\CID{1823} **不安定・発散 (Unstable):** 正の実部を持つ固有値が存在します。軍拡競争が暴走・発散する傾向にあります。")
    else:
        st.warning("\CID{244} **臨界状態 (Center/Saddle):** システムが安定と不安定の境界にあります。")

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
    if np.isinf(val) or np.isnan(val):
        return "発散 (測定不能)"
    return f"{val:.2e}" if abs(val) >= 100000 else f"{val:.1f}"

# 【修正】メトリクス表示にNカ国の分岐を追加
met1, met2, met3, met4, met5 = st.columns(5)
met1.metric("経過時間", f"{t_max:.1f}")

if model_type in ["2カ国 線形モデル (基本)", "2カ国 非線形モデル (抑制項の導入)"]:
    met2.metric("A国の軍備量 (X)", format_large_val(final_x))
    met3.metric("B国の軍備量 (Y)", format_large_val(final_y))
    met4.empty()
elif model_type in ["3カ国 線形モデル (同盟と敵対)", "3カ国 非線形モデル (抑制項の導入)"]:
    met2.metric("A国の軍備量 (X)", format_large_val(final_x))
    met3.metric("B国の軍備量 (Y)", format_large_val(final_y))
    met4.metric("C国の軍備量 (Z)", format_large_val(final_z))
else:
    met2.metric("国1の軍備量", format_large_val(final_X[0]))
    met3.metric("国2の軍備量", format_large_val(final_X[1]))
    if N >= 3:
        met4.metric("国3の軍備量", format_large_val(final_X[2]))
    else:
        met4.empty()

if is_linear_stable:
    met5.success("収束 (安定)")
else:
    met5.error("発散 (暴走)")

if model_type in ["2カ国 線形モデル (基本)", "2カ国 非線形モデル (抑制項の導入)"]:
    if model_type == "2カ国 線形モデル (基本)":
        if is_linear_stable:
            st.info("\CID{239} 平衡状態に向かっています。 ($b_1 b_2 - a_1 a_2 > 0$)")
        elif det == 0:
            st.warning("\CID{244} 安定と暴走の境界線上にあります。 ($b_1 b_2 - a_1 a_2 = 0$)")
        else:
            st.error("\CID{1823} 制御不能な軍拡競争が発生しています。 ($b_1 b_2 - a_1 a_2 < 0$)")
    else:
        st.info("\CID{2349} 経済的摩擦（非線形項）が強力なブレーキとして働くことで、軍備の無限の暴走が抑え込まれ、新たな均衡点（一定の水準）へと収束します。")
elif model_type in ["3カ国 線形モデル (同盟と敵対)", "3カ国 非線形モデル (抑制項の導入)"]:
    if model_type == "3カ国 線形モデル (同盟と敵対)":
        if is_linear_stable:
            st.info("\CID{239} 3カ国間のパワーバランスがとれ、平衡状態に向かっています。")
        else:
            st.error("\CID{1823} 同盟と敵対の連鎖により、制御不能な軍拡競争が発生しています。")
    else:
        st.info("\CID{2349} 経済的摩擦（非線形項）が強力なブレーキとして働くことで、軍備の無限の暴走が抑え込まれ、新たな均衡点（一定の水準）へと収束します。")
    st.info("\CID{657} A国、B国、C国の3カ国による相互作用モデルです。対抗係数をマイナスに設定することで「同盟 (相手の軍備増強が自国の安心につながる)」を表現できます。")
else:
    if N >= 3:
        st.info(f"\CID{654} {N}カ国による一般化・多国間相互作用モデルです。サイドバーの表を直接編集して、複雑な同盟ネットワークを構築できます。\n\n※相図は{N}次元空間をそのまま可視化できないため、代表して「国1・国2・国3」の部分空間に射影した3D軌跡を表示しています。")
    else:
        st.info(f"\CID{654} {N}カ国による一般化・多国間相互作用モデルです。サイドバーの表を直接編集して、複雑な同盟ネットワークを構築できます。")

# ==========================================
# 5. メイン画面：棒グラフの描画
# ==========================================
fig_bar, ax_bar = plt.subplots(figsize=(8, 1.2))

if model_type in ["2カ国 線形モデル (基本)", "2カ国 非線形モデル (抑制項の導入)"]:
    bar_x = final_x if np.isfinite(final_x) else 0
    bar_y = final_y if np.isfinite(final_y) else 0
    ax_bar.bar(["A国", "B国"], [bar_x, bar_y], color=['#4169E1', '#CD5C5C'], width=0.4)
    ax_bar.set_ylim(0, max(100, max(bar_x, bar_y) * 1.2))
elif model_type in ["3カ国 線形モデル (同盟と敵対)", "3カ国 非線形モデル (抑制項の導入)"]:
    bar_x = final_x if np.isfinite(final_x) else 0
    bar_y = final_y if np.isfinite(final_y) else 0
    bar_z = final_z if np.isfinite(final_z) else 0
    ax_bar.bar(["A国", "B国", "C国"], [bar_x, bar_y, bar_z], color=['#4169E1', '#CD5C5C', '#2ecc71'], width=0.4)
    ax_bar.set_ylim(0, max(100, max(bar_x, bar_y, bar_z) * 1.2))
else:
    bar_X = [x if np.isfinite(x) else 0 for x in final_X]
    colors = plt.cm.tab10.colors
    ax_bar.bar([f"国{i+1}" for i in range(N)], bar_X, color=[colors[i % 10] for i in range(N)], width=0.4)
    ax_bar.set_ylim(0, max(100, max(bar_X) * 1.2) if len(bar_X) > 0 else 100)

ax_bar.set_ylabel("軍備量")
st.pyplot(fig_bar)

# ==========================================
# 6 & 7. メイン画面：グラフを1つのキャンバスに統合
# ==========================================
def get_valid_max(arr):
    valid = arr[np.isfinite(arr)]
    return np.max(valid) if len(valid) > 0 else 0

def get_valid_min(arr):
    valid = arr[np.isfinite(arr)]
    return np.min(valid) if len(valid) > 0 else 0

if model_type in ["2カ国 線形モデル (基本)", "2カ国 非線形モデル (抑制項の導入)"]:
    max_val_all = max(100, max(get_valid_max(x_val), get_valid_max(y_val)) * 1.1)
    min_val_all = min(0, min(get_valid_min(x_val), get_valid_min(y_val)) * 1.1)
    if model_type == "2カ国 線形モデル (基本)" and is_linear_stable:
        if np.isfinite(eq_x) and np.isfinite(eq_y):
            max_val_all = max(max_val_all, max(eq_x, eq_y) * 1.2)
            min_val_all = min(min_val_all, min(eq_x, eq_y) * 1.2)
elif model_type in ["3カ国 線形モデル (同盟と敵対)", "3カ国 非線形モデル (抑制項の導入)"]:
    max_val_all = max(100, max(get_valid_max(x_val), get_valid_max(y_val), get_valid_max(z_val)) * 1.1)
    min_val_all = min(0, min(get_valid_min(x_val), get_valid_min(y_val), get_valid_min(z_val)) * 1.1)
    if is_linear_stable and eq_x is not None:
        if np.isfinite(eq_x) and np.isfinite(eq_y) and np.isfinite(eq_z):
            max_val_all = max(max_val_all, max(eq_x, eq_y, eq_z) * 1.2)
            min_val_all = min(min_val_all, min(eq_x, eq_y, eq_z) * 1.2)
else:
    max_val_all = max(100, get_valid_max(X_val) * 1.1)
    min_val_all = min(0, get_valid_min(X_val) * 1.1)
    if is_linear_stable and eq_X is not None:
        eq_valid = [v for v in eq_X if np.isfinite(v)]
        if len(eq_valid) > 0:
            max_val_all = max(max_val_all, max(eq_valid) * 1.2)
            min_val_all = min(min_val_all, min(eq_valid) * 1.2)

fig_main = plt.figure(figsize=(12, 5))

# 左側：時間発展グラフ
ax_line = fig_main.add_subplot(1, 2, 1)
ax_line.set_title("軍備拡張の時間発展", fontsize=14, pad=10)

if model_type in ["2カ国 線形モデル (基本)", "2カ国 非線形モデル (抑制項の導入)"]:
    ax_line.plot(t_val, x_val, label='A国 (X)', color='#4169E1', linewidth=2)
    ax_line.plot(t_val, y_val, label='B国 (Y)', color='#CD5C5C', linewidth=2)
    if is_linear_stable and eq_x is not None:
        if np.isfinite(eq_x) and np.isfinite(eq_y):
            ax_line.axhline(y=eq_x, color='#4169E1', linestyle=':', alpha=0.5)
            ax_line.axhline(y=eq_y, color='#CD5C5C', linestyle=':', alpha=0.5)
elif model_type in ["3カ国 線形モデル (同盟と敵対)", "3カ国 非線形モデル (抑制項の導入)"]:
    ax_line.plot(t_val, x_val, label='A国 (X)', color='#4169E1', linewidth=2)
    ax_line.plot(t_val, y_val, label='B国 (Y)', color='#CD5C5C', linewidth=2)
    ax_line.plot(t_val, z_val, label='C国 (Z)', color='#2ecc71', linewidth=2)
    if is_linear_stable and eq_x is not None:
        if np.isfinite(eq_x) and np.isfinite(eq_y) and np.isfinite(eq_z):
            ax_line.axhline(y=eq_x, color='#4169E1', linestyle=':', alpha=0.5)
            ax_line.axhline(y=eq_y, color='#CD5C5C', linestyle=':', alpha=0.5)
            ax_line.axhline(y=eq_z, color='#2ecc71', linestyle=':', alpha=0.5)
else:
    colors = plt.cm.tab10.colors
    for i in range(N):
        ax_line.plot(t_val, X_val[:, i], label=f'国{i+1}', color=colors[i % 10], linewidth=2)
    if is_linear_stable and eq_X is not None:
        for i in range(N):
            if np.isfinite(eq_X[i]):
                ax_line.axhline(y=eq_X[i], color=colors[i % 10], linestyle=':', alpha=0.5)

ax_line.axhline(y=0, color='black', linewidth=1.0, alpha=0.5)
ax_line.set_xlabel("時間 ($t$)")
ax_line.set_ylabel("軍備量")
ax_line.set_xlim(0, max(t_max, 50))
ax_line.set_ylim(min_val_all, max_val_all)
ax_line.legend(loc='upper left')
ax_line.grid(True, linestyle='--', alpha=0.7)

# 右側：相図
if model_type in ["2カ国 線形モデル (基本)", "2カ国 非線形モデル (抑制項の導入)"]:
    ax_phase = fig_main.add_subplot(1, 2, 2)
    ax_phase.set_title("相図", fontsize=14, pad=10)
    ax_phase.plot(x_val, y_val, color='#9b59b6', linewidth=2.5, label='軌跡')
    ax_phase.scatter([x_val[0]], [y_val[0]], color='#2ecc71', s=100, zorder=5, label='初期状態')
    ax_phase.scatter([final_x], [final_y], color='#3498db', s=100, zorder=5, label='最終状態 (t)')

    if model_type == "2カ国 線形モデル (基本)" and is_linear_stable:
        if np.isfinite(eq_x) and np.isfinite(eq_y):
            ax_phase.scatter([eq_x], [eq_y], color='#e74c3c', marker='*', s=300, zorder=6, label='平衡点 (解析解)')

    ax_phase.set_xlabel("A国の軍備量 (X)")
    ax_phase.set_ylabel("B国の軍備量 (Y)")
    ax_phase.set_xlim(min_val_all, max_val_all)
    ax_phase.set_ylim(min_val_all, max_val_all)
    ax_phase.legend(loc='upper left')
    ax_phase.grid(True, linestyle='--', alpha=0.7)

elif model_type in ["3カ国 線形モデル (同盟と敵対)", "3カ国 非線形モデル (抑制項の導入)"]:
    ax_phase = fig_main.add_subplot(1, 2, 2, projection='3d')
    ax_phase.set_title("3D相図 (空間軌跡)", fontsize=14, pad=10)
    ax_phase.plot(x_val, y_val, z_val, color='#9b59b6', linewidth=2.5, label='軌跡')
    ax_phase.scatter([x_val[0]], [y_val[0]], [z_val[0]], color='#2ecc71', s=100, label='初期状態')
    ax_phase.scatter([final_x], [final_y], [final_z], color='#3498db', s=100, label='最終状態 (t)')
    
    if is_linear_stable and eq_x is not None:
        if np.isfinite(eq_x) and np.isfinite(eq_y) and np.isfinite(eq_z):
            ax_phase.scatter([eq_x], [eq_y], [eq_z], color='#e74c3c', marker='*', s=300, label='平衡点 (解析解)')
    
    ax_phase.set_xlabel("A国 (X)")
    ax_phase.set_ylabel("B国 (Y)")
    ax_phase.set_zlabel("C国 (Z)", labelpad=10)
    ax_phase.set_xlim(min_val_all, max_val_all)
    ax_phase.set_ylim(min_val_all, max_val_all)
    ax_phase.set_zlim(min_val_all, max_val_all)
    ax_phase.legend(loc='upper left')

else:
    # Nカ国モデルの場合の相図（最初の最大3カ国を描画）
    if N >= 3:
        ax_phase = fig_main.add_subplot(1, 2, 2, projection='3d')
        ax_phase.set_title("3D相図 (国1〜国3への射影)", fontsize=14, pad=10)
        ax_phase.plot(X_val[:,0], X_val[:,1], X_val[:,2], color='#9b59b6', linewidth=2.5, label='軌跡')
        ax_phase.scatter([X_val[0,0]], [X_val[0,1]], [X_val[0,2]], color='#2ecc71', s=100, label='初期状態')
        ax_phase.scatter([final_X[0]], [final_X[1]], [final_X[2]], color='#3498db', s=100, label='最終状態 (t)')
        if is_linear_stable and eq_X is not None:
            if np.isfinite(eq_X[0]) and np.isfinite(eq_X[1]) and np.isfinite(eq_X[2]):
                ax_phase.scatter([eq_X[0]], [eq_X[1]], [eq_X[2]], color='#e74c3c', marker='*', s=300, label='平衡点 (解析解)')
        ax_phase.set_xlabel("国1"); ax_phase.set_ylabel("国2"); ax_phase.set_zlabel("国3", labelpad=10)
        ax_phase.set_xlim(min_val_all, max_val_all); ax_phase.set_ylim(min_val_all, max_val_all); ax_phase.set_zlim(min_val_all, max_val_all)
    else:
        ax_phase = fig_main.add_subplot(1, 2, 2)
        ax_phase.set_title("相図", fontsize=14, pad=10)
        ax_phase.plot(X_val[:,0], X_val[:,1], color='#9b59b6', linewidth=2.5, label='軌跡')
        ax_phase.scatter([X_val[0,0]], [X_val[0,1]], color='#2ecc71', s=100, zorder=5, label='初期状態')
        ax_phase.scatter([final_X[0]], [final_X[1]], color='#3498db', s=100, zorder=5, label='最終状態 (t)')
        if is_linear_stable and eq_X is not None:
            if np.isfinite(eq_X[0]) and np.isfinite(eq_X[1]):
                ax_phase.scatter([eq_X[0]], [eq_X[1]], color='#e74c3c', marker='*', s=300, zorder=6, label='平衡点 (解析解)')
        ax_phase.set_xlabel("国1"); ax_phase.set_ylabel("国2")
        ax_phase.set_xlim(min_val_all, max_val_all); ax_phase.set_ylim(min_val_all, max_val_all)
        ax_phase.grid(True, linestyle='--', alpha=0.7)
    
    ax_phase.legend(loc='upper left')

fig_main.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.15, wspace=0.3)
st.pyplot(fig_main)