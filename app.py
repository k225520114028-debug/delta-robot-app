import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib
import json
import pandas as pd
import os

# Cấu hình giao diện di động
st.set_page_config(page_title="Delta Robot AI Agent", layout="wide", initial_sidebar_state="collapsed")

# Cache mô hình AI vào bộ nhớ đệm để chạy cực nhanh trên điện thoại
@st.cache_resource
def load_model_from_file(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        return None

@st.cache_data
def load_json_config(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# Tự động tìm kiếm các file .joblib và .json trong thư mục ứng dụng
all_files = os.listdir('.')
joblib_files = [f for f in all_files if f.endswith('.joblib')]
json_files = [f for f in all_files if f.endswith('.json') and f != 'package.json']

# Phân loại mô hình Thuận và Ngược dựa trên tên file
fk_model_path = None
ik_model_path = None

for f in joblib_files:
    f_lower = f.lower()
    if 'thuan' in f_lower or 'fk' in f_lower or 'forward' in f_lower:
        fk_model_path = f
    elif 'nguoc' in f_lower or 'ik' in f_lower or 'inverse' in f_lower:
        ik_model_path = f

# Nếu không phân loại được theo tên, tự động gán file đầu tiên tìm thấy
if not fk_model_path and len(joblib_files) > 0:
    fk_model_path = joblib_files[0]
if not ik_model_path and len(joblib_files) > 1:
    ik_model_path = joblib_files[1]

st.title("🤖 AI Agent - Động học Robot Delta (3RUU)")

tab1, tab2 = st.tabs(["🔄 Bài toán Động học Thuận", "↩️ Bài toán Động học Ngược"])

# --- TAB 1: BÀI TOÁN THUẬN ---
with tab1:
    st.header("1. Cấu hình & Thông số hình học")
    
    model_fk = None
    if fk_model_path:
        model_fk = load_model_from_file(fk_model_path)
        if model_fk is not None:
            st.success(f"✅ Đã đóng gói sẵn Mô hình Thuận: `{fk_model_path}`")
    
    if model_fk is None:
        st.warning("⚠️ Chưa tìm thấy file `.joblib` trong ứng dụng. Bạn có thể nạp thủ công:")
        uploaded_fk = st.file_uploader("Nạp Mô hình Thuận (.joblib)", type=["joblib"], key="fk_upload")
        if uploaded_fk:
            model_fk = joblib.load(uploaded_fk)

    # Hiển thị thông số từ file JSON nếu có
    if json_files:
        config_data = load_json_config(json_files[0])
        if config_data:
            with st.expander(f"📄 Xem cấu hình từ file JSON tích hợp ({json_files[0]})"):
                st.json(config_data)

    col_r1, col_r2, col_a, col_b = st.columns(4)
    R = col_r1.number_input("R (mm)", value=100.0, step=1.0)
    r = col_r2.number_input("r (mm)", value=50.0, step=1.0)
    a = col_a.number_input("a (mm)", value=250.0, step=1.0)
    b = col_b.number_input("b (mm)", value=440.0, step=1.0)

    st.subheader("2. Phạm vi góc khớp q11..q33 (Độ)")
    q_ranges = {}
    default_ranges = [
        ("q11", -45.0, 90.0), ("q12", 0.0079, 829.3170), ("q13", 10.3123, 2747.4852),
        ("q21", -45.0, 90.0), ("q22", 0.0011, 838.6188), ("q23", 0.1505, 1481.5487),
        ("q31", -45.0, 90.0), ("q32", 0, 1324.7812), ("q33", 0.1648, 3647.0087)
    ]
    
    cols = st.columns(3)
    for idx, (name, d_min, d_max) in enumerate(default_ranges):
        with cols[idx % 3]:
            st.caption(f"**Góc {name}**")
            c_min, c_max = st.columns(2)
            q_min = c_min.number_input(f"Min {name}", value=d_min, key=f"min_{name}", format="%.4f")
            q_max = c_max.number_input(f"Max {name}", value=d_max, key=f"max_{name}", format="%.4f")
            q_ranges[name] = (q_min, q_max)

    n_samples = st.slider("Số lượng điểm mẫu (Samples)", min_value=500, max_value=20000, value=3000, step=500)

    if st.button("🚀 Chạy Bài toán Thuận & Vẽ Pointcloud 3D", use_container_width=True):
        st.info("Đang tính toán vùng làm việc...")
        
        q_samples = {}
        for name, (q_min, q_max) in q_ranges.items():
            q_samples[name] = np.random.uniform(np.radians(q_min), np.radians(q_max), n_samples)

        points_xyz = []
        if model_fk is not None:
            try:
                X_in = np.column_stack([
                    np.full(n_samples, R), np.full(n_samples, r),
                    np.full(n_samples, a), np.full(n_samples, b),
                    q_samples["q11"], q_samples["q12"], q_samples["q13"],
                    q_samples["q21"], q_samples["q22"], q_samples["q23"],
                    q_samples["q31"], q_samples["q32"], q_samples["q33"]
                ])
                points_xyz = model_fk.predict(X_in)
                st.success("Dự đoán tọa độ bằng AI Agent tích hợp thành công!")
            except Exception as e:
                st.warning(f"Lỗi khi chạy mô hình AI: {e}. Tự động chuyển sang tính toán hình học.")
                points_xyz = np.array([
                    [(R - r) + a * np.cos(q_samples["q11"][i]) + b * np.cos(q_samples["q13"][i]) * np.cos(q_samples["q12"][i]),
                     b * np.sin(q_samples["q13"][i]) * np.cos(q_samples["q12"][i]),
                     a * np.sin(q_samples["q11"][i]) + b * np.sin(q_samples["q12"][i])]
                    for i in range(n_samples)
                ])
        else:
            points_xyz = np.array([
                [(R - r) + a * np.cos(q_samples["q11"][i]) + b * np.cos(q_samples["q13"][i]) * np.cos(q_samples["q12"][i]),
                 b * np.sin(q_samples["q13"][i]) * np.cos(q_samples["q12"][i]),
                 a * np.sin(q_samples["q11"][i]) + b * np.sin(q_samples["q12"][i])]
                for i in range(n_samples)
            ])

        # Tải file CSV
        df_out = pd.DataFrame(points_xyz, columns=["Px", "Py", "Pz"])
        csv_bytes = df_out.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Tải file CSV kết quả P(x,y,z)", csv_bytes, "delta_workspace.csv", "text/csv")

        # Đồ thị 3D xoay/zoom cảm ứng
        fig = go.Figure(data=[go.Scatter3d(
            x=points_xyz[:, 0], y=points_xyz[:, 1], z=points_xyz[:, 2],
            mode='markers',
            marker=dict(size=2, color=points_xyz[:, 2], colorscale='Viridis', opacity=0.7)
        )])
        fig.update_layout(scene=dict(
            xaxis_title='Px (mm)', yaxis_title='Py (mm)', zaxis_title='Pz (mm)'
        ), margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig, use_container_width=True)

# --- TAB 2: BÀI TOÁN NGƯỢC ---
with tab2:
    st.header("Tính toán 9 góc khớp từ tọa độ P(x, y, z)")
    
    model_ik = None
    if ik_model_path:
        model_ik = load_model_from_file(ik_model_path)
        if model_ik is not None:
            st.success(f"✅ Đã đóng gói sẵn Mô hình Ngược: `{ik_model_path}`")

    if model_ik is None:
        st.warning("⚠️ Chưa tìm thấy file `.joblib` cho bài toán ngược. Bạn có thể nạp thủ công:")
        uploaded_ik = st.file_uploader("Nạp Mô hình Ngược (.joblib)", type=["joblib"], key="ik_upload")
        if uploaded_ik:
            model_ik = joblib.load(uploaded_ik)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    px = col_p1.number_input("Px (mm)", value=0.0, format="%.4f")
    py = col_p2.number_input("Py (mm)", value=0.0, format="%.4f")
    pz = col_p3.number_input("Pz (mm)", value=350.0, format="%.4f")

    if st.button("🎯 Tính toán Động học Ngược", use_container_width=True):
        if model_ik is not None:
            try:
                X_ik = np.array([[R, r, a, b, px, py, pz]])
                q_pred = model_ik.predict(X_ik)[0]
                
                st.subheader("Kết quả 9 góc khớp (Độ):")
                angles_names_list = ["q11", "q12", "q13", "q21", "q22", "q23", "q31", "q32", "q33"]
                res_cols = st.columns(3)
                for idx, q_name in enumerate(angles_names_list):
                    deg_val = np.degrees(q_pred[idx])
                    res_cols[idx % 3].metric(label=q_name, value=f"{deg_val:.4f}°")
            except Exception as e:
                st.error(f"Lỗi tính toán IK: {e}")
        else:
            st.error("Chưa có mô hình AI bài toán ngược để tính toán!")