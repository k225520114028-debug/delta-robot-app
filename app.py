import streamlit as st
import numpy as np
import plotly.graph_objects as go
import joblib
import pandas as pd
import io

# Cấu hình trang hiển thị di động
st.set_page_config(page_title="Delta Robot AI Agent", layout="wide", initial_sidebar_state="collapsed")

st.title("🤖 AI Agent - Động học Robot Delta (3RUU)")

# Mở 2 Tab chức năng cho bài toán Thuận và Ngược
tab1, tab2 = st.tabs(["🔄 Bài toán Động học Thuận", "↩️ Bài toán Động học Ngược"])

# --- TAB 1: BÀI TOÁN THUẬN ---
with tab1:
    st.header("1. Cấu hình & Thông số hình học")
    
    col_model, col_geo = st.columns(2)
    with col_model:
        uploaded_fk_joblib = st.file_uploader("Nạp Mô hình Thuận (.joblib)", type=["joblib"], key="fk_joblib")
    with col_geo:
        col_r1, col_r2 = st.columns(2)
        R = col_r1.number_input("R (mm)", value=100.0, step=1.0)
        r = col_r2.number_input("r (mm)", value=50.0, step=1.0)
        a = col_r1.number_input("a (mm)", value=250.0, step=1.0)
        b = col_r2.number_input("b (mm)", value=440.0, step=1.0)

    st.subheader("2. Phạm vi góc khớp q11..q33 (Độ)")
    q_ranges = {}
    default_ranges = [
        ("q11", -45.0, 90.0), ("q12", -30.0, 30.0), ("q13", 150.0, 210.0),
        ("q21", -45.0, 90.0), ("q22", -30.0, 30.0), ("q23", 150.0, 210.0),
        ("q31", -45.0, 90.0), ("q32", -30.0, 30.0), ("q33", 150.0, 210.0)
    ]
    
    # Tạo lưới nhập dải góc khớp 3x3
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
        
        # Sinh mẫu ngẫu nhiên dạng Radian
        q_samples = {}
        for name, (q_min, q_max) in q_ranges.items():
            q_samples[name] = np.random.uniform(np.radians(q_min), np.radians(q_max), n_samples)

        points_xyz = []
        if uploaded_fk_joblib is not None:
            try:
                model = joblib.load(uploaded_fk_joblib)
                X_in = np.column_stack([
                    np.full(n_samples, R), np.full(n_samples, r),
                    np.full(n_samples, a), np.full(n_samples, b),
                    q_samples["q11"], q_samples["q12"], q_samples["q13"],
                    q_samples["q21"], q_samples["q22"], q_samples["q23"],
                    q_samples["q31"], q_samples["q32"], q_samples["q33"]
                ])
                points_xyz = model.predict(X_in)
                st.success("Dự đoán tọa độ bằng AI thành công!")
            except Exception as e:
                st.warning(f"Không thể chạy Joblib: {e}. Tự động tính bằng công thức hình học.")
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

        # Xuất file CSV cho người dùng tải về điện thoại
        df_out = pd.DataFrame(points_xyz, columns=["Px", "Py", "Pz"])
        csv_bytes = df_out.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Tải file CSV kết quả P(x,y,z)", csv_bytes, "delta_workspace.csv", "text/csv")

        # Biểu diễn đồ thị 3D PointCloud (Xoay/Zoom bằng tay mượt mà trên điện thoại)
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
    uploaded_ik_joblib = st.file_uploader("Nạp Mô hình Ngược (.joblib)", type=["joblib"], key="ik_joblib")
    
    col_p1, col_p2, col_p3 = st.columns(3)
    px = col_p1.number_input("Px (mm)", value=0.0, format="%.4f")
    py = col_p2.number_input("Py (mm)", value=0.0, format="%.4f")
    pz = col_p3.number_input("Pz (mm)", value=350.0, format="%.4f")

    if st.button("🎯 Tính toán Động học Ngược", use_container_width=True):
        if uploaded_ik_joblib is not None:
            try:
                model_ik = joblib.load(uploaded_ik_joblib)
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
            st.info("Vui lòng nạp file `.joblib` bài toán ngược để dự đoán.")