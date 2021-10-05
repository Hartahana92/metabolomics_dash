import streamlit as st
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def main():
    page = st.sidebar.selectbox(
        "Выбери опцию",
        [
            "Домой",
            "График изменений концентрации вещества во времени",
            "Матрица корреляций",
            "Box-plot",
            "Гистограмма"
        ], )
    if page == "Домой":
        # st.header("Data Application")
        st.balloons()
    elif page == "График изменений концентрации вещества во времени":
        st.header("График изменений концентрации вещества во времени")
        simple_plot(df=df)
    elif page == "Матрица корреляций":
        st.header("Матрица корреляций")
        corr_matrix(df=df)
    elif page == "Box-plot":
        st.header("Box-plot")
        plot_boxes(df=df)
    elif page == "Гистограмма":
        st.header("Гистограмма")
        hist(df=df)


def corr_matrix(df):
    fig = plt.figure(figsize=(20, 20))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    st.pyplot(fig)


def simple_plot(df):
    df2 = df[df['Class'] == 'blank']
    df1 = df[df['Class'] == 'test']
    df1.drop('Class', axis='columns', inplace=True)
    df2.drop('Class', axis='columns', inplace=True)
    df1 = df1.transpose()
    df1['mean'] = df1.mean(numeric_only=True, axis=1)
    df1['Std'] = df1.std(numeric_only=True, axis=1)
    df2 = df2.transpose()
    df2['mean'] = df2.mean(numeric_only=True, axis=1)
    df2['Std'] = df2.std(numeric_only=True, axis=1)
    df.drop('Class', axis='columns', inplace=True)
    time = df.columns
    error2 = df2['Std']
    error1 = df1['Std']
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.errorbar(time, df2['mean'], yerr=error2, fmt='-o', mec='grey', ecolor='grey', mfc='grey', c='grey', capsize=3,
                label='blank')
    ax.errorbar(time, df1['mean'], yerr=error1, fmt='-o', mec='black', ecolor='black', mfc='black', c='black',
                capsize=3, label='test')
    ax.set_ylabel('Concentration')
    ax.set_xlabel('Time [min]')
    plt.legend(loc='upper left')
    st.pyplot(fig)


def plot_boxes(df):
    df_1 = df[0:6]
    df_2 = df[6:13]
    for i in range(df.shape[1]):
        fig, ax = plt.subplots(ncols=1)
        a = df_1[df_1.columns[i]]
        b = df_2[df_2.columns[i]]

        df_result = pd.DataFrame({'test': a, 'blank': b})
        boxplot = df_result.boxplot(column=['test', 'blank'], grid=False, ax=ax)
        ax.set_title(df_1.columns[i])
        st.pyplot(fig)


def hist(df):
    df1 = df[df['Group'] == 1]
    df2 = df[df['Group'] == 0]
    df3 = df1[metabolite]
    df4 = df2[metabolite]
    pyplot.hist(df3, alpha=0.5, label='Больные')
    pyplot.hist(df4, alpha=0.5, label='Здоровые')
    pyplot.legend(loc='upper right')
    pyplot.show()
    st.pyplot()


if __name__ == "__main__":
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Рисовалка всяких картинок для статей 🧐')
    st.sidebar.title('Что рисовать будем?')
    uploaded_file = st.file_uploader("Загрузим файл")
    metabolite = st.text_input("Введите название метаболита")
    name = st
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)

    main()
