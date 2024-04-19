import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Загрузка данных
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Обучение модели логистической регрессии
def train_model(X, y, normalization_method):
    if normalization_method == 'StandardScaler':
        scaler = StandardScaler()
    elif normalization_method == 'MinMaxScaler':
        scaler = MinMaxScaler()
    else:
        scaler = None
    
    if scaler is not None:
        X_normalized = scaler.fit_transform(X)
    else:
        X_normalized = X
        
    model = LogisticRegression(max_iter=1000)
    model.fit(X_normalized, y)
    return model

# Отображение результатов весов
def show_weights(feature_names, weights):
    st.write("Результаты логистической регрессии:")
    for feature, weight in zip(feature_names, weights[0]):
        st.write(f"{feature}: {weight}")

# Отображение scatter plot
def show_scatter_plot_with_decision_boundary(data, x_column, y_column, target_column, model):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x=x_column, y=y_column, hue=target_column, palette='coolwarm', ax=ax, marker='*', s=250)
    betas = model.coef_[0]
    beta_0 = model.intercept_[0]
    beta_1, beta_2 = betas[0], betas[1]
    x_values = np.array(ax.get_xlim())
    y_values = (- beta_0 - beta_1 * x_values) / beta_2
    ax.plot(x_values, y_values, linestyle='--', color='black', label='Decision Boundary')

    plt.title(f"Scatter plot: {x_column} vs {y_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    st.pyplot(fig)

# Функция для нормализации данных
def normalize_data(X, method):
    if method == "StandardScaler":
        scaler = StandardScaler()
    elif method == "MinMaxScaler":
        scaler = MinMaxScaler()
    else:
        return X
    return scaler.fit_transform(X)

# Главная функция
def main():
    
    
    #st.title("Логистическая регрессия")
    st.markdown(
        """
        <div style="background-color:#f0f0f0;padding:10px;border-radius:10px;">
            <h1 style="color:#333;text-align:center;">📊 Логистическая регрессия 📈</h1>
        </div>
        """,
        unsafe_allow_html=True
    )


    # Загрузка файла CSV
    uploaded_file = st.file_uploader("Загрузите файл CSV", type="csv")
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Отражение данных:")
        st.write(data.head())

        # Выбор фичей и таргета
        selected_columns = st.multiselect("Выберите столбцы для фичей (X)", data.columns)
        target_column = st.selectbox("Выберите столбец для таргета (y)", data.columns)
        
        # Выбор метода нормализации
        normalization_method = st.selectbox("Выберите метод нормализации", ['StandardScaler', 'MinMaxScaler', 'Нет, спасибо!'])

        # Применение нормализации к выбранным столбцам
        if normalization_method != 'Нет, спасибо!':
            columns_to_normalize = st.multiselect("Выберите столбцы для нормализации", selected_columns)
            data_to_normalize = data[columns_to_normalize]
            if normalization_method == 'StandardScaler':
                scaler = StandardScaler()
            elif normalization_method == 'MinMaxScaler':
                scaler = MinMaxScaler()
            data_normalized = scaler.fit_transform(data_to_normalize)
            data[columns_to_normalize] = data_normalized
        
        X = data[selected_columns]
        y = data[target_column]

        # Обучение модели
        model = train_model(X, y, normalization_method)

        # Отображение результатов
        show_weights(selected_columns, model.coef_)
        
        # Построение scatter plot с разделяющей прямой
        if len(selected_columns) >= 2:
            x_column = st.selectbox("Выберите столбец для оси X", selected_columns)
            y_column = st.selectbox("Выберите столбец для оси Y", selected_columns)
            show_scatter_plot_with_decision_boundary(data, x_column, y_column, target_column, model)
        st.balloons()
        st.success("Ура! Спасибо что воспользовались приложением! 🚀")
        st.markdown('<img src="https://media1.giphy.com/media/1ofR3QioNy264/giphy.gif">', unsafe_allow_html=True)

if __name__ == "__main__":
    main()