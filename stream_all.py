import streamlit as st
import matplotlib.pyplot as plt
import math
import pandas as pd

# تنظیمات صفحه
st.set_page_config(page_title="محاسبه پارامترهای قوس", layout="wide")

# تنظیم راست‌چین و اسکرول افقی برای جداول + استایل لوگوها و پلات
st.markdown("""
<style>
body, .stText, .stNumberInput, .stSelectbox, .stButton, .stDataFrame, .stSubheader, .stWrite {
    direction: rtl;
    text-align: right;
}
.stSelectbox > div > div > div {
    max-width: 200px;
}
.stDataFrame, .stTable {
    direction: rtl;
}
.stDataFrame table, .stTable table {
    direction: rtl;
    width: 100%;
    display: block;
    overflow-x: auto;
    white-space: nowrap;
}
.stDataFrame th, .stDataFrame td, .stTable th, .stTable td {
    text-align: right !important;
    unicode-bidi: embed;
}


</style>
""", unsafe_allow_html=True)

# تابع محاسبه جدول پیاده‌سازی


def calculate_implementation_table(stations, segments):
    impl_data = []
    for i in range(len(stations) - 1):
        stn_prev = stations[i]
        stn_curr = stations[i + 1]
        delta = stn_curr - stn_prev
        for seg in segments:
            seg_start, seg_end, R = seg
            if stn_prev >= seg_start and stn_curr <= seg_end:
                delta_theta = delta / R
                delta_theta_deg = math.degrees(delta_theta)
                chord = 2 * R * math.sin(delta_theta / 2)
                impl_data.append((
                    round(stn_prev, 3),
                    round(stn_curr, 3),
                    round(delta, 3),
                    round(delta_theta_deg, 3),
                    round(chord, 3)
                ))
                break
    return pd.DataFrame(impl_data, columns=["از ایستگاه", "تا ایستگاه", "فاصله (m)", "زاویه انحراف (°)", "طول وتر (m)"])

# تابع محاسبات برای قوس ساده


def calculate_simple_curve(R, D_deg, KP, interval):
    D = math.radians(D_deg)
    L = R * D
    T = R * math.tan(D/2)
    C = 2 * R * math.sin(D/2)
    A = KP - T
    B = A + L
    params = {
        "طول مماس (T)": round(T, 3),
        "طول وتر قوس (C)": round(C, 3),
        "کیلومتر شروع قوس (A)": round(A, 3),
        "طول قوس (L)": round(L, 3),
        "کیلومتر انتهای قوس (B)": round(B, 3)
    }
    stations = [A]
    first = math.ceil(A / interval) * interval
    cur = first
    while cur < B:
        stations.append(cur)
        cur += interval
    stations.append(B)
    station_data = []
    prev = stations[0]
    for s in stations:
        delta = s - prev if s != stations[0] else 0
        station_data.append((round(s, 3), round(delta, 3)))
        prev = s
    segments = [(A, B, R)]
    impl_table = calculate_implementation_table(stations, segments)
    plot_data = {
        "R": R,
        "D": D,
        "D_deg": D_deg,
        "T": T,
        "A": A,
        "B": B,
        "L": L,
        "stations": stations,
        "KP": KP,
        "interval": interval
    }
    return {"params": params, "station_data": station_data, "plot_data": plot_data, "impl_table": impl_table, "error": None}

# تابع محاسبات برای قوس مرکب


def calculate_compound_curve(KP, R1, D1_deg, R2, D2_deg, interval):
    D1 = math.radians(D1_deg)
    D2 = math.radians(D2_deg)
    L1 = R1 * D1
    L2 = R2 * D2
    T1 = R1 * math.tan(D1/2)
    T2 = R2 * math.tan(D2/2)
    T_shared = T1 + T2
    T_in = T1 + T_shared * math.sin(D2) / math.sin(D1 + D2)
    T_out = T2 + T_shared * math.sin(D1) / math.sin(D1 + D2)
    A = KP - T_in
    B = A + L1 + L2
    params = {
        "KP رأس": KP,
        "KP شروع (A)": round(A, 3),
        "L1 (m)": round(L1, 3),
        "L2 (m)": round(L2, 3),
        "L کل (m)": round(L1 + L2, 3),
        "Δ1 (°)": D1_deg,
        "Δ2 (°)": D2_deg,
        "Δ کل (°)": round(D1_deg + D2_deg, 3),
        "KP پایان (B)": round(B, 3),
        "T1 (m)": round(T1, 3),
        "T2 (m)": round(T2, 3),
        "T مشترک (m)": round(T_shared, 3),
        "مسیر ورودی (m)": round(T_in, 3),
        "مسیر خروجی (m)": round(T_out, 3)
    }
    stations = [A]
    first = math.ceil(A / interval) * interval
    pos = first
    while pos < B:
        stations.append(pos)
        pos += interval
    stations.append(B)
    station_data = []
    prev = stations[0]
    for s in stations:
        delta = s - prev if s != stations[0] else 0
        station_data.append((round(s, 3), round(delta, 3)))
        prev = s
    segments = [(A, A + L1, R1), (A + L1, B, R2)]
    impl_table = calculate_implementation_table(stations, segments)
    plot_data = {
        "R1": R1,
        "D1": D1,
        "D1_deg": D1_deg,
        "R2": R2,
        "D2": D2,
        "D2_deg": D2_deg,
        "A": A,
        "B": B,
        "L1": L1,
        "L2": L2,
        "stations": stations,
        "KP": KP,
        "interval": interval
    }
    return {"params": params, "station_data": station_data, "plot_data": plot_data, "impl_table": impl_table, "error": None}

# تابع محاسبات برای قوس معکوس


def calculate_reverse_curve(R, p, KP0, dKP):
    if p >= 2 * R:
        return {"params": None, "station_data": None, "plot_data": None, "impl_table": None, "error": "مقدار p باید کمتر از 2R باشد."}
    theta = math.acos(1 - p / (2 * R))
    L1 = R * theta
    L_tot = 2 * L1
    chord = 2 * R * math.sin(theta / 2)
    KP_end = KP0 + L_tot
    params = {
        "θ (°)": round(math.degrees(theta / 2), 3),
        "2θ (°)": round(math.degrees(theta), 3),
        "L هر قوس [m]": round(L1, 3),
        "L کل [m]": round(L_tot, 3),
        "C وتر هر قوس [m]": round(chord, 3),
        "KP شروع [m]": round(KP0, 3),
        "KP پایان [m]": round(KP_end, 3),
    }
    stations = [KP0]
    first = math.ceil(KP0 / dKP) * dKP
    if KP0 < first < KP_end:
        stations.append(first)
    x = first + dKP
    while x < KP_end:
        stations.append(x)
        x += dKP
    stations.append(KP_end)
    station_data = []
    prev = stations[0]
    for s in stations:
        delta = s - prev if s != stations[0] else 0
        station_data.append((round(s, 3), round(delta, 3)))
        prev = s
    segments = [(KP0, KP0 + L1, R), (KP0 + L1, KP_end, R)]
    impl_table = calculate_implementation_table(stations, segments)
    plot_data = {
        "R": R,
        "p": p,
        "KP0": KP0,
        "dKP": dKP,
        "theta": theta,
        "L1": L1,
        "L_tot": L_tot,
        "chord": chord,
        "KP_end": KP_end,
        "stations": stations
    }
    return {"params": params, "station_data": station_data, "plot_data": plot_data, "impl_table": impl_table, "error": None}

# تابع ترسیم برای قوس ساده


def plot_simple_curve(plot_data):
    fig, ax = plt.subplots(figsize=(7, 7))
    R = plot_data["R"]
    D = plot_data["D"]
    D_deg = plot_data["D_deg"]
    T = plot_data["T"]
    A = plot_data["A"]
    B = plot_data["B"]
    L = plot_data["L"]
    stations = plot_data["stations"]
    tangent_color = 'orange'
    radius_color = 'purple'
    cx, cy = 0, R
    thetas = [i/1000 * D for i in range(1001)]
    xs = [R * math.sin(t) for t in thetas]
    ys = [R * (1 - math.cos(t)) for t in thetas]
    ax.plot(xs, ys, color='blue', lw=2, label='Curve')
    x_pc, y_pc = 0, 0
    x_pt = R * math.sin(D)
    y_pt = R * (1 - math.cos(D))
    dir_pt = (math.cos(D), math.sin(D))
    t_int = -y_pt / dir_pt[1]
    int_x = x_pt + t_int * dir_pt[0]
    int_y = 0
    ax.plot([int_x, x_pc], [int_y, y_pc], color=tangent_color,
            linestyle='--', lw=2, label='Tangent')
    ax.plot([int_x, x_pt], [int_y, y_pt],
            color=tangent_color, linestyle='--', lw=2)
    ax.plot([cx, x_pc], [cy, y_pc], color=radius_color, lw=2, label='Radius')
    ax.plot([cx, x_pt], [cy, y_pt], color=radius_color, lw=2)
    start_ang = -(math.pi/2 - D)
    end_ang = -math.pi/2
    center_r = R/6
    theta_c = [start_ang + i*(end_ang-start_ang)/100 for i in range(101)]
    cx_arc = [cx + center_r*math.cos(th) for th in theta_c]
    cy_arc = [cy + center_r*math.sin(th) for th in theta_c]
    ax.plot(cx_arc, cy_arc, color='green', linestyle='--', lw=2)
    mid_ang = (start_ang + end_ang) / 2
    ax.text(cx + (center_r+R*0.02)*math.cos(mid_ang), cy + (center_r+R*0.02)*math.sin(mid_ang),
            f"Δ={round(D_deg, 2)}°", color='green', fontsize=12, fontweight='bold')
    for stn in stations:
        frac = (stn - A) / L * D
        xs_s = R * math.sin(frac)
        ys_s = R * (1 - math.cos(frac))
        ax.plot(xs_s, ys_s, 'ko')
        ax.text(xs_s, ys_s, f"{round(stn, 1)}", fontsize=8)
    ax.text(cx + R*0.1, cy - R*0.1, f"R={R}", fontsize=10, color=radius_color)
    ax.text((int_x+x_pc)/2, (int_y+y_pc)/2 - R*0.02,
            f"T={round(T, 3)}", fontsize=10, color=tangent_color)
    ax.text((int_x+x_pt)/2 + dir_pt[0]*0.02, (int_y+y_pt)/2 + dir_pt[1]
            * 0.02, f"T={round(T, 3)}", fontsize=10, color=tangent_color)
    ax.set_aspect('equal')
    ax.set_title('Simple Curve', fontsize=14)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    return fig

# تابع ترسیم برای قوس مرکب


def plot_compound_curve(plot_data):
    fig, ax = plt.subplots(figsize=(8, 8))
    R1 = plot_data["R1"]
    D1 = plot_data["D1"]
    D1_deg = plot_data["D1_deg"]
    R2 = plot_data["R2"]
    D2 = plot_data["D2"]
    D2_deg = plot_data["D2_deg"]
    A = plot_data["A"]
    B = plot_data["B"]
    L1 = plot_data["L1"]
    L2 = plot_data["L2"]
    stations = plot_data["stations"]
    cx1, cy1 = 0, 0
    theta1 = [math.pi/2 - D1 + D1*i/1000 for i in range(1001)]
    x1 = [cx1 + R1*math.cos(t) for t in theta1]
    y1 = [cy1 + R1*math.sin(t) for t in theta1]
    vx, vy = x1[-1], y1[-1]
    rx, ry = cx1 - vx, cy1 - vy
    norm = math.hypot(rx, ry)
    ux, uy = rx/norm, ry/norm
    cx2 = vx + R2*ux
    cy2 = vy + R2*uy
    start_ang = math.atan2(vy - cy2, vx - cx2)
    theta2 = [start_ang + D2*i/1000 for i in range(1001)]
    x2 = [cx2 + R2*math.cos(t) for t in theta2]
    y2 = [cy2 + R2*math.sin(t) for t in theta2]
    ax.plot(x1, y1, '-b', lw=2, label='1st curve')
    ax.plot(x2, y2, '-r', lw=2, label='2nd curve')
    ax.plot(cx1, cy1, 'go', label='C1')
    ax.plot(cx2, cy2, 'mo', label='C2')
    ax.plot(vx, vy, 'ko')
    pcx, pcy = x1[0], y1[0]
    ax.plot([cx1, pcx], [cy1, pcy], '--g')
    ax.plot([cx1, vx], [cy1, vy], 'g')
    tpx, tpy = x2[-1], y2[-1]
    ax.plot([cx2, vx], [cy2, vy], '--m')
    ax.plot([cx2, tpx], [cy2, tpy], '--m')
    a1 = math.atan2(pcy - cy1, pcx - cx1)
    b1 = math.atan2(vy - cy1, vx - cx1)
    arc1 = [a1 + (b1 - a1)*i/50 for i in range(51)]
    ax1 = [cx1 + (R1/6)*math.cos(a) for a in arc1]
    ay1 = [cy1 + (R1/6)*math.sin(a) for a in arc1]
    ax.plot(ax1, ay1, 'g')
    ma1 = (a1 + b1)/2
    ax.text(cx1 + (R1/6 + 10)*math.cos(ma1), cy1 + (R1/6 + 10)
            * math.sin(ma1), f"Δ1={D1_deg}°", color='green')
    a2 = start_ang
    b2 = math.atan2(tpy - cy2, tpx - cx2)
    if b2 < a2:
        b2 += 2*math.pi
    arc2 = [a2 + (b2 - a2)*i/50 for i in range(51)]
    ax2 = [cx2 + (R2/6)*math.cos(a) for a in arc2]
    ay2 = [cy2 + (R2/6)*math.sin(a) for a in arc2]
    ax.plot(ax2, ay2, 'm')
    ma2 = (a2 + b2)/2
    ax.text(cx2 + (R2/6 + 10)*math.cos(ma2), cy2 + (R2/6 + 10)
            * math.sin(ma2), f"Δ2={D2_deg}°", color='purple')
    for s in stations:
        frac = (s - A)/(L1 + L2)*(D1 + D2)
        if frac <= D1:
            xx = cx1 + R1*math.cos(math.pi/2 - D1 + frac)
            yy = cy1 + R1*math.sin(math.pi/2 - D1 + frac)
        else:
            f2 = frac - D1
            xx = cx2 + R2*math.cos(start_ang + f2)
            yy = cy2 + R2*math.sin(start_ang + f2)
        ax.plot(xx, yy, 'ks')
        ax.text(xx, yy, f"{round(s, 1)}", fontsize=8)
    ax.set_aspect('equal')
    ax.set_title("Compound Curve", fontsize=14)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    return fig

# تابع ترسیم برای قوس معکوس


def plot_reverse_curve(plot_data):
    fig, ax = plt.subplots(figsize=(5, 5))
    R = plot_data["R"]
    p = plot_data["p"]
    KP0 = plot_data["KP0"]
    dKP = plot_data["dKP"]
    theta = plot_data["theta"]
    L1 = plot_data["L1"]
    L_tot = plot_data["L_tot"]
    chord = plot_data["chord"]
    KP_end = plot_data["KP_end"]
    stations = plot_data["stations"]
    top_tan = 0
    bot_tan = -p
    xmax = KP0 + L_tot + R / 5
    ax.plot([KP0 - R / 5, xmax], [top_tan, top_tan],
            '--', lw=1.2, color='orange')
    ax.plot([KP0 - R / 5, xmax], [bot_tan, bot_tan],
            '--', lw=1.2, color='orange')
    T1 = (KP0, top_tan)
    T2 = (KP0 + 2 * R * math.sin(theta), bot_tan)
    ax.plot(*T1, 'rs', ms=6)
    ax.text(T1[0], T1[1], ' T1', color='red', va='bottom')
    ax.plot(*T2, 'rs', ms=6)
    ax.text(T2[0], T2[1], ' T2', color='red', va='top')
    O1 = (KP0, top_tan - R)
    O2 = (KP0 + 2 * R * math.sin(theta), -R + 2 * R * math.cos(theta))
    ax.plot(*O1, 'go', ms=6)
    ax.text(O1[0], O1[1], ' O1', color='green', va='top')
    ax.plot(*O2, 'mo', ms=6)
    ax.text(O2[0], O2[1], ' O2', color='magenta', va='bottom')
    ax.plot([O1[0], O2[0]], [O1[1], O2[1]], '--k', lw=1)
    ax.plot([O1[0], T1[0]], [O1[1], T1[1]], '--', color='green')
    ax.plot([O2[0], T2[0]], [O2[1], T2[1]], '--', color='magenta')
    th1_start = math.pi / 2
    th1_end = math.pi / 2 - theta
    thetas1 = [th1_start + (th1_end - th1_start) * i / 200 for i in range(201)]
    x1 = [O1[0] + R * math.cos(t) for t in thetas1]
    y1 = [O1[1] + R * math.sin(t) for t in thetas1]
    ax.plot(x1, y1, '-b', lw=2)
    th2_start = 3 * math.pi / 2 - theta
    th2_end = 3 * math.pi / 2
    th2_start = 3 * math.pi / 2 - theta
    th2_end = 3 * math.pi / 2
    thetas2 = [th2_start + (th2_end - th2_start) * i / 200 for i in range(201)]
    x2 = [O2[0] + R * math.cos(t) for t in thetas2]
    y2 = [O2[1] + R * math.sin(t) for t in thetas2]
    ax.plot(x2, y2, '-g', lw=2)
    for s in stations:
        s_rel = s - KP0
        if s_rel <= L1:
            phi = s_rel / R
            t = math.pi / 2 - phi
            xs = O1[0] + R * math.cos(t)
            ys = O1[1] + R * math.sin(t)
        else:
            s2 = s_rel - L1
            phi = s2 / R
            t = (3 * math.pi / 2 - theta) + phi
            xs = O2[0] + R * math.cos(t)
            ys = O2[1] + R * math.sin(t)
        ax.plot(xs, ys, 'k.', ms=5)
        ax.text(xs, ys, f"{int(round(s, 0))}",
                fontsize=5, ha='center', va='bottom')
    t_pi = math.pi / 2 - theta
    x_pi = O1[0] + R * math.cos(t_pi)
    y_pi = O1[1] + R * math.sin(t_pi)
    ax.plot(x_pi, y_pi, 'ro', ms=8)
    ax.text(x_pi, y_pi, ' PI', color='red',
            fontsize=10, va='bottom', ha='left')
    ax.set_aspect('equal')
    ax.set_title("Reverse Curve", fontsize=14)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    return fig


# برنامه اصلی
st.title("محاسبه پارامترهای قوس")

# منوی کشویی در مرکز
col1, col2, col3 = st.columns([1.5, 1, 1.5])
with col2:
    curve_type = st.selectbox(
        "انتخاب نوع قوس", ["قوس ساده", "قوس مرکب", "قوس معکوس"])

# ورودی‌ها بر اساس نوع قوس
_, input_col, _ = st.columns([0.1, 0.8, 0.1])
with input_col:
    if curve_type == "قوس ساده":
        R = st.number_input("شعاع قوس (R) [m]", value=100.0)
        D_deg = st.number_input("زاویه انحراف (D) [°]", value=65.0)
        KP = st.number_input("کیلومتر راس قوس (KP) [m]", value=1320.15)
        interval = st.number_input("فاصله بین ایستگاه‌ها [m]", value=30.0)
    elif curve_type == "قوس مرکب":
        KP = st.number_input("کیلومتر راس قوس (KP) [m]", value=1187.939)
        R1 = st.number_input("شعاع قوس اول (R1) [m]", value=200.0)
        D1_deg = st.number_input("زاویه انحراف اول (D1) [°]", value=30.0)
        R2 = st.number_input("شعاع قوس دوم (R2) [m]", value=300.0)
        D2_deg = st.number_input("زاویه انحراف دوم (D2) [°]", value=40.0)
        interval = st.number_input("فاصله بین ایستگاه‌ها [m]", value=50.0)
    elif curve_type == "قوس معکوس":
        R = st.number_input("شعاع قوس R [m]", value=200.0)
        p = st.number_input("فاصله بین مماس‌ها p [m]", value=50.0)
        KP0 = st.number_input("KP شروع [m]", value=1000.0)
        dKP = st.number_input("فاصله ایستگاه‌ها [m]", value=30.0)

# دکمه محاسبه
_, button_col, _ = st.columns([0.1, 0.8, 0.1])
with button_col:
    if st.button("محاسبه"):
        try:
            if curve_type == "قوس ساده":
                result = calculate_simple_curve(R, D_deg, KP, interval)
            elif curve_type == "قوس مرکب":
                result = calculate_compound_curve(
                    KP, R1, D1_deg, R2, D2_deg, interval)
            elif curve_type == "قوس معکوس":
                result = calculate_reverse_curve(R, p, KP0, dKP)

            if result['error'] is not None:
                st.error(result['error'])
            else:
                # نمایش پارامترها
                _, content_col, _ = st.columns([0.1, 0.8, 0.1])
                with content_col:
                    params_df = pd.DataFrame(
                        list(result['params'].items()), columns=["پارامتر", "مقدار"])
                    st.subheader("پارامترهای محاسبه شده")
                    st.dataframe(
                        params_df, use_container_width=True, hide_index=True)

                    # نمایش ایستگاه‌ها
                    station_df = pd.DataFrame(result['station_data'], columns=[
                                              "کیلومتراژ (متر)", "فاصله از ایستگاه قبلی (متر)"])
                    st.subheader("ایستگاه‌ها")
                    st.dataframe(
                        station_df, use_container_width=True, hide_index=True)

                    # نمایش جدول پیاده‌سازی
                    st.subheader("جدول پیاده‌سازی")
                    st.dataframe(result['impl_table'],
                                 use_container_width=True, hide_index=True)

                    # نمایش نمودار
                    st.subheader("نمودار قوس")
                    if curve_type == "قوس ساده":
                        fig = plot_simple_curve(result['plot_data'])
                    elif curve_type == "قوس مرکب":
                        fig = plot_compound_curve(result['plot_data'])
                    elif curve_type == "قوس معکوس":
                        fig = plot_reverse_curve(result['plot_data'])
                    st.pyplot(fig)
        except ValueError:
            st.error("لطفاً مقادیر معتبر وارد کنید.")
