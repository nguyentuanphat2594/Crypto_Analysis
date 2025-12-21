import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as stats
import time
import traceback
from hyperopt import hp, fmin, tpe, Trials, pyll, STATUS_OK
from hyperopt.fmin import generate_trials_to_calculate
from functools import partial
from backtesting import Backtest, Strategy
from sklearn.preprocessing import RobustScaler
from scipy.stats import ttest_1samp, wilcoxon

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1220;
        color: #e5e7eb;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("📈 Ứng dụng phân tích dữ liệu và xây dựng mô hình dự đoán giá Crypto")
page = st.sidebar.selectbox(
    "Chọn chức năng",
    ["Load data và Thống kê chung", "EDA", "Model"]
)

if page == "Load data và Thống kê chung":
    st.header("📊 Load data và Thống kê chung")
    # Load data
    choice = st.radio(
        "Chọn nguồn dữ liệu",
        ["Sample sẵn có", "Upload dữ liệu của bạn"]
    )
    if choice == "Sample sẵn có":
        data = pd.read_csv("ETHUSDT.csv")
    else:
        """***TỆP CẦN CÓ CÁC CỘT: timestamp, open, high, low, close, volume theo thứ tự.***"""
        data = st.file_uploader("Upload your CSV file", type=["csv", "xlsx"])
        if data is not None:
            if data.name.endswith('.csv'):
                data = pd.read_csv(data)
            elif data.name.endswith('.xlsx'):
                data = pd.read_excel(data)

    if data is not None:
        try:
            st.subheader("🔍 Kiểm tra chất lượng dữ liệu")
            st.write(f'**Số hàng trùng lắp:**')
            st.write(f'**{data.duplicated().sum()}** hàng')
            data.rename(columns={'timestamp': 'Timestamp', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume' }, inplace=True)
            data['Timestamp'] = pd.to_datetime(data['Timestamp'])
            data.set_index('Timestamp', inplace=True)
            data.sort_index(inplace=True)
            dup_ts = data.index.duplicated().sum()
            st.write(f"**Số timestamp trùng lặp:**")
            st.write(f"{dup_ts}")

            st.subheader('Thông tin chung')
            st.write(f"#### Data Shape:")
            st.write(f"**{data.shape[0]} rows**, **{data.shape[1]} columns**")
            st.write(f"#### Date range:")
            st.write(f"**{data.index.min()}** to **{data.index.max()}**")
            st.write("#### Missing Values")
            st.dataframe(data.isna().sum().to_frame("Missing Count")) # Dùng to_frame để hiển vì output là series
            st.write(F"#### Thống kê mô tả")                             
            st.dataframe(data.describe().T) # DÙng T vì output là dataframe

        except Exception as e:
            st.error(f"Vui lòng đặt tên cột đúng định dạng, thứ tự: timestamp, open, high, low, close, volume")
            st.stop()

    # Lưu data dùng chung
    st.session_state['data'] = data
    

elif page == "EDA":
    st.header("📈 EDA")

    data = st.session_state.data
    if data is None:
        st.warning("Vui lòng tải dữ liệu trong mục 'Thống kê chung' trước khi sử dụng chức năng này.")
        st.stop()

    # Tính, trực sai phân bậc 1 giá đóng cửa
    diff = data['Close'].diff()
    window = 240
    min_periods = 100
    
    rolling_mean = diff.rolling(window, min_periods=min_periods).mean()
    rolling_std  = diff.rolling(window, min_periods=min_periods).std()

    st.subheader('Xu hướng cục bộ và mức độ biến động của ETH (30 phút)')
    fig, ax = plt.subplots(2, 1, figsize=(20,10), sharex=True)

    # Rolling mean
    ax[0].plot(rolling_mean, label='Rolling Mean (Diff)')
    ax[0].axhline(0, linestyle='--', alpha=0.5)
    ax[0].set_title('Rolling Mean of First-order Differencing')
    ax[0].legend()

    # Rolling std
    ax[1].plot(rolling_std, label='Rolling Std (Volatility)')
    ax[1].set_title('Rolling Volatility (Std of Diff)')
    ax[1].legend()

    st.pyplot(fig)

    close_diff = diff

    rolling_mean = close_diff.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = close_diff.rolling(window=window, min_periods=min_periods).std()

    z_scores = (close_diff - rolling_mean)/rolling_std

    thresold = 3
    thresold1 = 4
    thresold2 = 5
    data_diff_filtered = close_diff.copy()
    mask = np.abs(z_scores) > thresold
    mask1 = np.abs(z_scores) > thresold1
    mask2 = np.abs(z_scores) > thresold2

    len_diff_greater3 = len(data_diff_filtered.loc[mask])
    rate_diff_greater3 = round(len_diff_greater3 / len(close_diff) * 100,2)
    def rate(rate):
        if rate <= 5:
            return 'Thấp'
        elif 5 < rate < 10:
            return 'Trung bình'
        else:
            return 'Cao'
    st.markdown(f"""
    **Tóm tắt thống kê (sai phân bậc 1):**
    - Trung bình lợi suất: **{diff.mean()*100:.4f}%**
    - Độ lệch chuẩn lợi suất: **{diff.std():.2f}**
    - Độ biến động lớn nhất (rolling std): **{rolling_std.max():.2f}**
    - Sai phân bậc 1 giá đóng cửa vượt ngưỡng **{thresold}**: **{len_diff_greater3}** (gần **{rate_diff_greater3}%** lượng data) -> ***{rate(rate_diff_greater3)}***
    """)

    return_dist = close_diff

    # Vẽ biểu đồ Histogram
    st.subheader('Biểu đồ Histogram phân phối lợi nhuận theo tần suất')
    plt.figure(figsize=(10,5))
    plt.hist(return_dist, bins=100)
    plt.title(f"Biểu đồ Histogram thể hiện phân phối lợi nhuận theo tần suất\nSkew = {return_dist.skew():.2f}")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    st.pyplot(plt)
    def rate_iqr(iqr):
        if iqr < 0.005:
            return "Biến động thấp, phân phối tập trung"
        elif iqr < 0.015:
            return "Biến động trung bình"
        else:
            return "Biến động cao, rủi ro đuôi lớn"

    def rate_q1(q1):
        if q1 > -0.002:
            return "Đuôi âm nông, downside risk thấp"
        elif q1 > -0.01:
            return "Downside risk trung bình"
        else:
            return "Đuôi âm dày, rủi ro giảm mạnh"
    def rate_q3(q3):
        if q3 < 0.002:
            return "Biên độ tăng yếu"
        elif q3 < 0.01:
            return "Upside trung bình"
        else:
            return "Upside mạnh, xuất hiện các nhịp tăng lớn"
    q1 = return_dist.quantile(0.25)
    q3 = return_dist.quantile(0.75)
    iqr = q3 - q1

    st.markdown(f"""
    **Rủi ro & cơ hội đuôi phân phối (lợi suất):**
    - Q1 (25%): **{q1:.2f}** → *{rate_q1(q1)}*
    - Q3 (75%): **{q3:.2f}** → *{rate_q3(q3)}*
    - IQR: **{iqr:.2f}** → *{rate_iqr(iqr)}*
    """)

    # Vẽ biểu đồ Boxplot
    st.subheader('Biểu đồ thể hiện giá trị ngoại lai của lợi nhuận')
    plt.figure(figsize=(6,5))
    plt.boxplot(return_dist[1:], vert=True)
    plt.title(f"Biểu đồ boxplot xem các giá trị ngoại lai của lợi nhuận\nKurtosis = {return_dist.kurtosis():.2f}")
    plt.ylabel("Return")
    st.pyplot(plt)
    def rate_kurtosis(kurt):
        if kurt < 0:
            return "Phân phối bẹt, ít đuôi dày, tail risk thấp"
        elif kurt < 3:
            return "Phân phối gần chuẩn, tail risk trung bình"
        elif kurt < 7:
            return "Phân phối đuôi dày, thường xuyên xuất hiện biến động lớn"
        else:
            return "Đuôi rất dày, rủi ro cực đoan cao (fat-tail risk)"
    def rate_outlier_ratio(outlier_ratio):
        if outlier_ratio < 1:
            return "Rất ít ngoại lai, biến động tương đối ổn định"
        elif outlier_ratio < 5:
            return "Có một số ngoại lai, xuất hiện shock ngắn hạn"
        elif outlier_ratio < 10:
            return "Nhiều ngoại lai, thị trường biến động mạnh"
        else:
            return "Ngoại lai dày đặc, rủi ro biến động cực đoan"
    q1 = return_dist.quantile(0.25)
    q3 = return_dist.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = return_dist[(return_dist < lower_bound) | (return_dist > upper_bound)]
    outlier_ratio = len(outliers) / len(return_dist) * 100
    kurt = return_dist.kurtosis()

    st.markdown(f"""
    **Phân tích ngoại lai & tail risk (Boxplot):**
    - Kurtosis: **{kurt:.2f}** → *{rate_kurtosis(kurt)}*
    - Tỷ lệ ngoại lai: **{outlier_ratio:.2f}%** → *{rate_outlier_ratio(outlier_ratio)}*
    - Biên dưới (Lower bound): **{lower_bound:.2f}**
    - Biên trên (Upper bound): **{upper_bound:.2f}**

    Boxplot cho thấy sự xuất hiện của nhiều giá trị ngoại lai, phản ánh các cú biến động bất thường trong lợi suất.
    """)





elif page == "Model":
    st.header("📈 Xây dựng mô hình dự đoán xu hướng bằng thuật toán RandomForest")

    data = st.session_state.data
    if data is None:
        st.warning("Vui lòng tải dữ liệu trong mục 'Thống kê chung' trước khi sử dụng chức năng này.")
        st.stop()
    # Chuẩn bị các biến đặc trưng cơ bản
    df = data.copy()
    df['Return'] = df['Close'].pct_change()

    df['Daily'] = df.index.day
    df['Weekday'] = df.index.weekday

    df['Return_2'] = df['Return'].shift(2)

    df['Upper_shadow'] = (df['High'] - df['Close']) / df['Close']
    df['Lower_shadow'] = (df['Close'] - df['Low']) / df['Close']

    df['High_gap'] = (df['High'].shift(1) - df['Close']) / df['Close']
    df['Low_gap'] = (df['Close'] - df['Low'].shift(1)) / df['Close']

    df['vp_ratio_1'] = (df['Volume'].shift(1)) / (df['Close'].shift(1))

    features = ['Daily','Weekday','Return_2',
            'Upper_shadow','Lower_shadow',
            'High_gap', 'Low_gap',
            'vp_ratio_1','Return']

    corr = df[features].corr()

    st.subheader("Biểu đồ tương quan các biến cơ bản")
    plt.figure(figsize=(10,8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Tương quan các biến cơ bản vs lợi nhuận")
    st.pyplot(plt)

    # Tìm các đặc trưng có tương quan thấp so với return
    low_corr = corr['Return'].abs() < 0.05
    remove_low_corr = list(low_corr[low_corr].index)
    df = df.drop(remove_low_corr, axis=1)
    st.write(f'Đã loại bỏ các biến cơ bản có tương quan thấp với Return: {remove_low_corr}')

    # Chuẩn bị các biến đặc trưng chỉ báo 
    #1. RSI
    window = 14
    delta = df['Close'].diff()

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    # giữ nguyên index của df
    gain = pd.Series(gain, index=df.index)
    loss = pd.Series(loss, index=df.index)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()

    rs = avg_gain / avg_loss
    df['RSI14'] = 100 - (100 / (1 + rs))

    # 2. MACD 
    ema6 = df['Close'].ewm(span=6, adjust=False).mean()
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()

    df['MACD'] = ema6 - ema20
    df['Signal'] = df['MACD'].ewm(span=4, adjust=False).mean()
    df['MACD_HIST'] = df['MACD'] - df['Signal']

    # 3. ATR(14) normalized
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift(1))
    low_close = abs(df['Low'] - df['Close'].shift(1))

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR14'] = tr.ewm(span=14, adjust=False).mean()
    df['ATR_NORM'] = df['ATR14'] / df['Close']

    # 4. Volume Spike Relative Ratio

    df['Vol_shock'] = (df['Volume'] - df['Volume'].rolling(4).mean()) / df['Volume'].rolling(48).std()

    # 5. Spike-Weighted ATR
    df['TR'] = np.maximum(df['High'] - df['Low'],
                    np.maximum(abs(df['High'] - df['Close'].shift(1)),
                            abs(df['Low'] - df['Close'].shift(1))))

    # ATR cơ bản
    df['ATR4'] = df['TR'].rolling(4).mean()

    # Spike Factor = TR hiện tại cao bao nhiêu lần so với TR trung bình
    df['Spike_factor'] = df['TR'] / df['TR'].rolling(48).mean()

    # Spike-Weighted ATR
    df['SW_ATR'] = df['ATR4'] * df['Spike_factor']

    df.drop(columns=['MACD', 'Signal', 'ATR14', 'TR', 'ATR4', 'Spike_factor'], errors='ignore', inplace=True)

    indicator_features = ['RSI14', 'MACD_HIST', 'ATR_NORM', 'Vol_shock', 'SW_ATR', 'Return']
    indicator_features_corr = df[indicator_features].corr()

    st.subheader("Biểu đồ tương quan các biến chỉ báo kỹ thuật")
    plt.figure(figsize=(10,8))
    sns.heatmap(indicator_features_corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Tương quan các biến chỉ báo kỹ thuật vs lợi nhuận")
    st.pyplot(plt)
    # Tìm các đặc trưng có tương quan thấp so với return
    low_indicator_corr = indicator_features_corr['Return'].abs() < 0.05
    remove_low_indicator_corr = list(low_indicator_corr[low_indicator_corr].index)
    df = df.drop(remove_low_indicator_corr, axis=1)
    st.write(f'Đã loại bỏ các biến chỉ báo kỹ thuật có tương quan thấp với Return: {remove_low_indicator_corr}')

    df = df.dropna()

    st.subheader("Bắt đầu chuẩn hóa dữ liệu với RobustScaler")
    # Chọn các đặc trưng huấn luyện cho mô hình nhằm dự báo
    features = ['Upper_shadow', 'Lower_shadow', 'High_gap', 'Low_gap',
                'RSI14', 'MACD_HIST', 'SW_ATR', 'Vol_shock']

    # Chuẩn hóa theo RobusScaler
    scaler = RobustScaler()
    df[features] = scaler.fit_transform(df[features])
    st.write("Dữ liệu sau khi chuẩn hóa:")
    st.dataframe(df[features].describe().loc[['max', 'min', 'mean']])

    # Hàm chiến lược mua bán
    class GeneralStrategy(Strategy):
        def init(self):
            pass
        def next(self):
            signal = self.data.position[-1]
            if signal == 1:
                # Mở long if signal = 1:
                if not self.position:
                    # đang đứng ngoài -> mua vào
                    self.buy() # Đóng long
            elif signal == -1:
                if self.position.is_long:
                    # đang cầm long -> đóng
                    self.position.close()
                    # signal = 0 > đứng ngoài, không làm gì
    
    
    # Hàm tính Sharpe Ratio
    def compute_sharpe(returns, window=17520):
        try:
            return np.sqrt(window) * (returns.mean() / returns.std())
        except:
            return np.nan
    
    # Hàm sinh vị thế vị thế giao dịch
    def find_position(df, paras):
        """Hàm tìm vị thế giao dịch"""
        # Chèn tham số
        try:
            n_estimators = int(paras["n_estimators"])
            max_depth = int(paras["max_depth"])
            min_samples_split = int(paras["min_samples_split"])
            min_samples_leaf = int(paras["min_samples_leaf"])
        except:
            n_estimators = int(paras[0])
            max_depth = int(paras[1])
            min_samples_split = int(paras[2])
            min_samples_leaf = int(paras[3])

        df_rf = df.copy()

        # Tạo nhãn xu hướng (Trend)
        threshold = 0.001  # 0.1% return
        df_rf["Trend"] = df_rf["Close"].shift(-1)/df_rf["Close"] - 1
        df_rf["Trend"] = df_rf["Trend"].apply(lambda x: 1 if x > threshold else -1)
        df_rf = df_rf.dropna(subset=["Trend"])

        #  Split data (75% train, 25% test)
        train_size = int(len(df_rf) * 0.75)
        df_train = df_rf.iloc[:train_size]


        # Các feature được chọn để chạy model
        features = ['Upper_shadow','Lower_shadow','High_gap','Low_gap',
                    'RSI14','MACD_HIST','SW_ATR','Vol_shock']

        X_train = df_train[features]
        y_train = df_train["Trend"]

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Dự đoán xác suất của từng lớp (tăng/giảm) cho mỗi quan sát
        proba = model.predict_proba(df_rf[features])
        classes = model.classes_

        idx_up = np.where(classes == 1)[0][0]     # Tìm vị trí tăng
        idx_down = np.where(classes == -1)[0][0]  # Tìm vị trí giảm

        prob_up = proba[:, idx_up]                # Xác suất tăng
        prob_down = proba[:, idx_down]            # Xác suất lệ giảm

        # Tính delta: độ chênh giữa xác suất tăng và giảm
        # Delta dương -> kỳ vọng tăng, Delta âm -> kỳ vọng giảm
        delta = prob_up - prob_down
        delta = pd.Series(delta, index=df_rf.index)

        # Làm mượt delta bằng EMA (Exponential Moving Average) với span=48
        # Giúp giảm nhiễu và tạo tín hiệu ổn định hơn cho việc vào/thoát lệnh
        delta_smooth = delta.ewm(span=8, adjust=False).mean()

        # Ngưỡng kì vọng
        entry_thr = 0.06  # Ngưỡng kỳ vọng để vào lệnh mua (delta_smooth vượt mức này)
        exit_thr = -0.06 # Ngưỡng kỳ vọng để thoát lệnh/bán (delta_smooth dưới mức này)

        # Tạo biến vị thế
        pos_event = pd.Series(0.0, index=df_rf.index)

        state = 0         # Tình trạng (chưa vào lệnh: 0, đang giữ lệnh: 1)
        hold = 0          # Số phiên đã giữ lệnh
        min_hold = 0     # Số phiên tối thiểu cần giữ lệnh

        # Duyệt qua từng thời điểm để xác định tín hiệu giao dịch
        for t in df_rf.index:
            d = delta_smooth.at[t]

            # Nếu chưa vào lệnh
            if state == 0:
                if d > entry_thr:           # Nếu kỳ vọng tăng vượt ngưỡng -> vào lệnh mua
                    pos_event.at[t] = 1.0
                    state = 1
                    hold = 0

            elif state == 1:
                hold += 1
                if (d < exit_thr) and hold >= min_hold: # Nếu kỳ vọng giảm vượt ngưỡng và đã giữ đủ phiên → thoát lệnh
                    pos_event.at[t] = -1.0
                    state = 0
                    hold = 0

        # Để tránh việc dùng dữ liệu tương lai (look-ahead bias)
        pos_event = pos_event.shift(1).fillna(0.0)

        # Đảm bảo khớp index của df gốc
        pos_full = pd.Series(0.0, index=df.index)
        pos_full.loc[pos_event.index] = pos_event

        return pos_full

    # Hàm backtesting
    def backtest(df, cash, commission, print_result=True):
        bt = Backtest(df,
                        GeneralStrategy,
                        cash=cash,
                        commission=commission,
                        trade_on_close=False,
                        exclusive_orders=True)

        stats = bt.run()
        stats = stats.to_frame()

        # Tính sharpe ratio
        stats.loc['Duration'] = len(df)
        returns = stats.loc['_equity_curve'][0]['Equity'].pct_change() * 100
        stats.loc['Sharpe Ratio'] = compute_sharpe(returns=returns, window=len(df))

        # Làm tròn số
        stats = stats.round(2)

        if print_result:
            st.write("==== Backtest Results ====")
            st.write(f"Return: {stats.loc['Return [%]'][0]}%")
            st.write(f"Sharpe Ratio: {stats.loc['Sharpe Ratio'][0]}")
            st.write(f"Max Drawdown: {stats.loc['Max. Drawdown [%]'][0]}%")
            st.write(f"CAGR: {stats.loc['Return (Ann.) [%]'][0]}%")
            st.write(f"Win Rate: {stats.loc['Win Rate [%]'][0]}%")
            st.write(f"Total Trades: {int(stats.loc['# Trades'][0])}")
            st.write(f"Profit Factor: {stats.loc['Profit Factor'][0]}")
            st.write("==========================")

        return bt, stats
    
    # Hàm đánh giá hiệu suất mô hình
    def score(paras, df, cash, commission, fitness, print_result=False):

        # Huấn luyện model
        position_series = find_position(df, paras)

        # Gán tín hiệu vào cột 'position'
        df['position'] = position_series

        # Gọi backtest
        bt, stats = backtest(df, cash, commission, print_result=print_result)

        if fitness == 'combo':
            sharpe = stats.loc['Sharpe Ratio'][0]
            cagr = stats.loc['Return (Ann.) [%]'][0]
            res = -(sharpe + 0.1 * cagr)              # 0.1 vì cagr có đơn vị %
        elif fitness == 'sharpe':
            res = -stats.loc['Sharpe Ratio'][0]
        elif fitness == 'cagr':
            res = -stats.loc['Return (Ann.) [%]'][0]
        return res

    fspace = {
        'n_estimators': hp.quniform('n_estimators', 100, 500, 5), # Số cây trong rừng
        'max_depth': hp.quniform('max_depth', 10, 35, 1), # Độ sâu tối đa của cây
        'min_samples_split': hp.quniform('min_samples_split', 5, 70, 1), # Số lượng mẫu tối thiểu để chia một node
        'min_samples_leaf': hp.quniform('min_samples_leaf', 5, 70, 1) # Số lượng mẫu tối thiểu tại node lá
    }

    cash = 1000000
    commission = 0.001
    max_evals = 20
    fitness = 'cagr'
    df_rf = df.copy()
    paras_best = None
    
    st.header("🚀 Tuning Parasmeter")
    mode1 = st.radio(
        "",
        ["Sử dụng tham số hệ thống (nhanh)", "Tuning Parasmeter (chậm)"]
    )

    if mode1 == "Sử dụng tham số hệ thống (nhanh)":
        if st.button("Sử dụng tham số đã được chọn trước cho Random Forest"):
            paras_best = {
                "max_depth" : 22,
                "min_samples_leaf" : 23,
                "min_samples_split" : 42,
                "n_estimators" : 120
            }
    else:
        # Parameters tuning
        if st.button("🚀 Bắt đầu quá trình Tuning"):
            # Tạo dict chứa không gian tham số
            fmin_objective = partial(score, df=df_rf, cash=cash, commission=commission, fitness=fitness)
            # Tối ưu hóa tham số
            st.write("Running Parameter Tuning for Random Forest...")
            paras_best = fmin(fn=fmin_objective, space=fspace, algo=tpe.suggest, max_evals=max_evals)
            st.write("Best Parameters:", paras_best)   
         
    if paras_best is not None:
        # Backtest với tham số tốt nhất
        st.write('\nBacktest on Test Data (with Best RF Parameters):')
        # Lấy nửa sau để test tham số vừa tìm được
        df_rf_test = df_rf.copy()

        position_test = find_position(df_rf_test, paras_best)

        df_rf_test['position'] = position_test

        # Thực hiện backtest
        bt_test, stats_test = backtest(df_rf_test, cash, commission)
        bt_test.plot(plot_width = 900, plot_volume=False, superimpose=False)



    st.header("🚀 Walk-Forward Optimization")

    # Walk - Forward Optimization
    cash = 1000000
    commission = 0.001
    max_evals = 15
    fitness = 'cagr'
    num_split = 5

    # Tạo hàm chia dữ liệu theo seq
    def split_data(df_split, seq):
        df_train = pd.concat([df_split[seq], df_split[seq+1]])
        df_test = pd.concat([df_split[seq+2], df_split[seq+3]])

        return df_train, df_test

    # Tạo hàm in hiệu suất
    def print_perf(period, fitness, loss):
        perf_df = pd.DataFrame()
        perf_df[f'{fitness}_{period}'] = [-round(loss, 2)]

        return perf_df

    # Tạo hàm chứa số mặc định cho RF-model
    def init_paras():
        return {
                'n_estimators': 50,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            }

    def mean_ttest(df, mean_val, col=None):
        """Hàm đo hiệu suất kiểm định t-test"""
        if col is None:
            statistic, pval = ttest_1samp(df, mean_val, alternative='greater')
        else:
            statistic, pval = ttest_1samp(df[col], mean_val, alternative='greater')

        # Kiểm định pvalue với giá trị 10%
        m1_greater_m2 = pval < 0.1

        # Tạo dict chứ kết quả
        results = {
            "pvalue": pval,
            f"\n{df} greater {mean_val}": m1_greater_m2,
        }

        return results

    def compare_means_ttest(df1, df2, col1=None, col2=None):
        """Hàm đo hiệu suất kiểm định wilcoxon"""
        if col1 is None or col2 is None:
            statistic, pval = wilcoxon(df1, df2, alternative='greater')
        else:
            statistic, pval = wilcoxon(df1[col1], df2[col2], alternative='greater')

        # Kiểm định pvalue với giá trị 10%
        m1_greater_m2 = pval < 0.1

        # Tạo dict chứa kết quả kiểm định
        results = {
            "pvalue": pval,
            f"\n{df1} greater than {df2}": m1_greater_m2,
        }

        return results

    if st.button("🚀 Bắt đầu quá trình Walk-Forward Optimization"):
        """Vì quá trình walk-forward khá lâu nên tiết kiệm thời gian sẽ chạy với số lượng num split =  6"""
        time_start = time.time()

        perf_total_train = pd.DataFrame()
        perf_total_test = pd.DataFrame()

        # Chia dữ liệu thành các phần
        df_split = np.array_split(df[:int(len(df_rf)*0.7)].copy(), num_split)
        paras_default = init_paras()

        st.write("\n Bắt đầu quá trình Walk-Forward \n")

        fmin_objective_train = partial(score, cash=cash,
                                    commission=commission, fitness=fitness)
        # Walk-Forward loop
        for seq in range(len(df_split) - 3):
            df_train, df_test = split_data(df_split, seq)
            seq += 1
            #TRAIN
            try:
                trials = generate_trials_to_calculate([init_paras()])
                paras_train = fmin(fn=partial(score, df=df_train, cash=cash, commission=commission, fitness=fitness),
                                    space=fspace, algo=tpe.suggest, max_evals=max_evals, trials=trials, show_progressbar=False)
                loss_train = score(paras_train, df_train, cash, commission, fitness)

            except Exception as e:
                st.write(f"Lỗi trong quá trình tối ưu hóa trainning: {e}")
                traceback.print_exc()
                paras_train = paras_default
                loss_train = score(paras_train, df_train, cash,
                                            commission, fitness, print_result=False)

            # Tính loss với tham số mặc định
            loss_train_default = score(paras_default, df_train, cash,
                                        commission, fitness)
            st.write(f"""
                ### [{seq}] Train RF
                - **Ngày:** {df_train.index[0]}
                - **Tham số train:** {paras_train}
                - **Sai số:** {-loss_train}
                """
                )

            # TEST
            try:
                paras_test = paras_train
                loss_test = score(paras_test, df_test, cash, commission, fitness)
                loss_test_default = score(paras_default, df_test, cash, commission, fitness)

                st.write(f"""
                        ### [{seq}] Test RF 
                        - **Ngày - giờ:** {df_test.index[0]} 
                        - **Sai số (tham số tối ưu) trên tập test:** {-loss_test} 
                        - **Sai số (tham số mặc định) trên tập test:** {loss_test_default}
                """)

                perf_train = print_perf('Train_RF', fitness, loss_train)
                perf_test = print_perf('Test_RF', fitness, loss_test)

                perf_train_default = print_perf('Train_RF_default', fitness, loss_train_default)
                perf_test_default = print_perf('Test_RF_default', fitness, loss_test_default)

                temp_total_train = pd.concat([perf_train, perf_train_default], axis=1)
                temp_total_test = pd.concat([perf_test, perf_test_default], axis=1)

                perf_total_train = pd.concat([perf_total_train, temp_total_train])
                perf_total_test = pd.concat([perf_total_test, temp_total_test])

            except Exception as e:
                st.write('Lỗi trong quá trình testing ở gian đoạn thứ {}'.format(seq))
                traceback.print_exc()
                perf_total_train_rf = pd.concat([perf_total_train_rf, pd.DataFrame(np.nan, index=[0], columns=['Return_train_RF', 'Return_train_default_RF'])])
                perf_total_test_rf = pd.concat([perf_total_test_rf, pd.DataFrame(np.nan, index=[0], columns=['Return_test_RF', 'Return_test_default_RF'])])

        st.subheader("Đánh giá mô hình sau khi Walk-Forward Optimization")
        st.write(f"Best paras: {paras_train}")
        threshold = 0.001
        df_rf["Trend"] = df_rf["Close"].shift(-1)/df_rf["Close"] - 1
        df_rf["Trend"] = df_rf["Trend"].apply(lambda x: 1 if x > threshold else -1)
        df_rf = df_rf.dropna(subset=["Trend"])

        X = df_rf[features]
        y = df_rf['Trend']

        model = RandomForestClassifier(
            n_estimators=int(paras_train['n_estimators']),
            max_depth=int(paras_train['max_depth']),
            min_samples_split=int(paras_train['min_samples_split']),
            min_samples_leaf=int(paras_train['min_samples_leaf']),
            random_state=42
        )

        model.fit(X, y)
        y_pred = model.predict(X)

        # Đánh giá
        st.subheader("=== Evaluation on Final Walk-Forward Test Segment ===")
        st.write(confusion_matrix(y, y_pred))
        st.write(classification_report(y, y_pred))
        st.write("Accuracy:", accuracy_score(y, y_pred))
        st.write("F1 Score:", f1_score(y, y_pred, average='weighted'))

        st.write(f"\nLength of train data: {len(df_train)}, Length of test data: {len(df_test)}")
        st.write("\nMean Performance (Train - RF):")
        st.write(round(perf_total_train.mean(),2))
        st.write("\nMean Performance (Test - RF):")
        st.write(round(perf_total_test.mean(),2))
        st.write('Running time (RF WF): %.2fs -- %.2fm' % (time.time() - time_start, (time.time() - time_start)/60))


        perf_total_train.reset_index(drop=True, inplace=True)
        perf_total_train.index.names = ['Period']
        perf_total_test.reset_index(drop=True, inplace=True)
        perf_total_test.index.names = ['Period']

        st.subheader("\nPerformance Summary (RF WF):")
        summary_df = pd.concat([perf_total_train, perf_total_test], axis=1)
        st.dataframe(summary_df)


        st.subheader('Kiểm định thống kê cho kết quả Walk-Forward RF')

        """Kiểm định t-test xem tỷ suất lợi nhuận trung bình trên tập test (đã tối ưu) có lớn hơn 0 không"""
        results_test_optimized = mean_ttest(perf_total_test.dropna(), 0, 'cagr_Test_RF')
        st.write(f"""
                 #### *RF Optimized Test Return Annual (vs 0):* 
                 {results_test_optimized}
                """)

        """Kiểm định t-test xem tỷ suất lợi nhuận trung bình trên tập test (mặc định) có lớn hơn 0 không"""
        results_test_default = mean_ttest(perf_total_test.dropna(), 0, 'cagr_Test_RF_default')
        st.write(f"""
                 #### *RF Default Test Return Annual (vs 0):* 
                 {results_test_default}
            """)

        """Kiểm định Wilcoxon xem lợi nhuận trung bình trên tập test (tối ưu) có lớn hơn lợi nhuận (mặc định) không"""
        results_test_comparison = compare_means_ttest(perf_total_test.dropna(), perf_total_test.dropna(), 'cagr_Test_RF', 'cagr_Test_RF_default')
        st.write(f"""
                 #### *RF Test Return Anuual (Optimized vs Default):*
                {results_test_comparison}
            """)

        # Chạy backtest trên tập kiểm định
        df_ETH = df.copy()

        # Lấy tín hiệu giao dịch từ model cuối
        position_series = find_position(df_ETH, paras_train)

        df_ETH['position'] = position_series

        df_ETH_test = df_ETH

        bt, stats = backtest(df_ETH_test, cash, commission)
        bt.plot(plot_width=900, plot_volume=False, superimpose=False)