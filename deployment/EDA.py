import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def app():
    # Load your dataset
    df = pd.read_csv('tabelEDA.csv')
    st.header('Stroke Prediction')
    st.subheader('Dataset Preview')
    st.write(df)

    # Sidebar options
    st.sidebar.header('Exploratory Data Analysis')
    eda_option = st.sidebar.selectbox('Select Analysis', ['Univariate', 'Bivariate', 'Multivariate'])

    # Title
    st.title('Exploratory Data Analysis (EDA)')
    
    colors = sns.color_palette("Set2")

    if eda_option == 'Univariate':
        st.header('Univariate Analysis')

        # Descriptive Summary for Numerical Data
        st.subheader('Numerical Data Descriptive Summary')
        st.write(df.describe(exclude='object').T)

        # Descriptive Summary for Categorical Data
        st.subheader('Categorical Data Descriptive Summary')
        st.write(df.describe(include='object').T)

        # Distribution of Stroke Disease
        st.subheader('Distribution of Stroke Disease')
        count_stroke = df['stroke'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(count_stroke, labels=['No Stroke', 'Stroke'], explode=(0.1, 0.0), autopct='%1.2f%%', colors=colors)
        ax.axis('equal')
        st.pyplot(fig)
        st.write('Dari visualisasi distribusi penyakit stroke, diketahui bahwa terdapat 4.87% pasien menderita penyakit stroke dan 95.13% pasien tidak menderita penyakit stroke. Secara umum, proporsi penderita stroke relatif kecil dalam dataset.')

        # Distribution of Patient Ages
        st.subheader('Distribution of Patient Ages')
        fig, ax = plt.subplots()
        sns.histplot(df['age'], kde=True, stat='density', discrete=True, color='coral', alpha=0.6, ax=ax)
        ax.set_title('Age Distribution')
        ax.grid(False)
        st.pyplot(fig)
        st.write('Berdasarkan visualisasi distribusi usia, diketahui bahwa persebaran usia pasien berada di 0 - 82 tahun dengan usia terbanyak berada di usia 78 tahun. Hal ini menunjukkan bahwa pentingnya faktor usia terhadap penyakit stroke, risiko menderita stroke cenderung meningkat seiring bertambahnya usia.')

        # Distribution of Blood Glucose Levels
        st.subheader('Distribution of Blood Glucose Levels')
        fig, ax = plt.subplots()
        sns.histplot(df['avg_glucose_level'], kde=True, stat='density', discrete=True, color='skyblue', alpha=0.6, ax=ax)
        ax.set_title('Distribution of Glucose Levels')
        ax.grid(False)
        st.pyplot(fig)
        st.write('Berdasarkan visualisasi distribusi kadar glukosa darah, diketahui bahwa kadar glukosa darah pasien berkisar antara 55.12 - 271.74 mg/dL. Pasien rata-rata memiliki kadar glukosa darah sebesar 106.14 mg/dL. Mayoritas pasien memiliki kadar glukosa yang normal (70 - 110 mg/dL) yang dapat dilihat kurva tinggi di kisaran batasan normal. Meskipun mayoritas pasien memiliki kadar glukosa normal, distribusi juga mencakup pada pasien dengan kadar glukosa di tidak normal. Ini menunjukkan adanya variasi dalam kadar glukosa darah di antara pasien.')

        # Distribution of BMI
        st.subheader('Distribution of Body Mass Index')
        fig, ax = plt.subplots()
        sns.histplot(df['bmi'], kde=True, stat='density', discrete=True, color='skyblue', alpha=0.6, ax=ax)
        ax.set_title('Distribution of Body Mass Index')
        ax.grid(False)
        st.pyplot(fig)
        st.write('Berdasarkan visualisasi distribusi BMI, diketahui bahwa BMI pasien berada di rentang 10.3 - 97.6 Kg/m².')

        # Distributed based on gender, hypertension, heart disease, marital status, smoking status and type of residence 
        st.subheader('Distribution based on gender, hypertension, heart disease, marital status, smoking status and type of residence')
        data = df[['gender', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type','smoking_status']]
        fig, axs = plt.subplots(3, 2, figsize=(15, 20))

        ind = 0
        for i, col in enumerate(data.columns):
            ax = axs[i // 2, i % 2]
            value_counts = data[col].value_counts()
            ax.pie(value_counts, labels=value_counts.index, autopct='%1.2f%%', colors=colors, startangle=90, wedgeprops=dict(edgecolor='w'))
            ax.axis('equal')
            ax.set_title(col)
        
        plt.suptitle('Distribution based on gender, hypertension, heart disease, marital status, smoking status and type of residence', fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        st.pyplot(fig)
        interpretation = """
        Berdasarkan visualisasi faktor risiko stroke, diketahui bahwa:
        - Mayoritas pasien berjenis kelamin perempuan sebesar 58.60% dibandingan dengan laki-laki sebesar 41.40%
        - Sekitar 90.25% pasien tidak memiliki riwayat hipertensi dan 9.75% pasien memiliki riwayat hipertensi
        - Sekitar 94.60% pasien tidak memiliki penyakit jantung dan 5.40% pasien memiliki penyakit jantung
        - Mayoritas pasien sudah menikah sebesar 65.63% sedangkan yang belum menikah sebesar 34.37%
        - Mayoritas pasien bertempat tinggal di perkotaan sebesar 50.81% dibandingkan di perdesaan sebesar 49.19%
        - Sekitar 37% pasien tidak merokok, 30.22% tidak diketahui status merokok pasien, 17.30% pasien sebelumnya pernah merokok, dan 15.44% pasien merokok.
        """
        st.write(interpretation)

        # Distribution of Work Types
        st.subheader('Distribution of Work Types')
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='work_type', palette=colors, ax=ax)
        ax.set_title('Work Types Distribution')
        ax.set_xlabel('Work Type')
        ax.set_ylabel('Count')
        ax.grid(False)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    if eda_option == 'Bivariate':
        st.header('Bivariate Analysis')

        # Risk Factors Influence on Stroke Disease
        st.subheader('Risk Factors Influence on Stroke Disease')
        data1 = df[['gender', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type', 'smoking_status', 'work_type']]
        fig, axs = plt.subplots(4, 2, figsize=(16, 16))

        for i, a in enumerate(data1):
            row = i // 2  
            col = i % 2   
            ax = sns.countplot(x=a, hue='stroke', data=df, palette=colors, ax=axs[row, col])
            ax.legend(title='Stroke Disease Status', labels=['No Stroke Disease', 'Stroke Disease'])

        st.pyplot(fig)
    

        # Relationship between Numerical Features and Stroke
        st.subheader('Relationship between Numerical Features and Stroke')
        features = df[['age', 'avg_glucose_level', 'bmi', 'stroke']]
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(features.corr(), cmap="vlag", annot=True, linewidths=.8, ax=ax)
        st.pyplot(fig)
        st.write('Interpretasi: Korelasi antara fitur numerik menunjukkan hubungan antara usia, kadar glukosa, dan BMI terhadap stroke.')

    if eda_option == 'Multivariate':
        st.header('Multivariate Analysis')

        # Stroke Disease by Marital Status and Gender
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.barplot(data=df, x='ever_married', y='stroke', hue='gender', palette=colors, ci=None, dodge=True, ax=axes[0])
        axes[0].set_title('Stroke Disease by Marital Status and Gender')
        axes[0].set_xlabel('Marital Status')
        axes[0].set_ylabel('Stroke')
        axes[0].legend(title='Gender', loc='upper left')
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)

        # Stroke Disease by Hypertension and Heart Disease
        sns.barplot(data=df, x='hypertension', y='stroke', hue='heart_disease', palette=colors, ci=None, ax=axes[1])
        axes[1].set_title('Stroke Disease by Hypertension and Heart Disease')
        axes[1].set_xlabel('Hypertension')
        axes[1].set_ylabel('Stroke')
        axes[1].legend(title='Heart Disease')
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(fig)
        st.write('Interpretasi: Hubungan antara status pernikahan dan gender, hipertensi dan penyakit jantung dengan penyakit stroke.')

        # Stroke Disease by Work Type and Residence Type
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        sns.barplot(data=df, x='work_type', y='stroke', hue='Residence_type', palette=colors, ci=None, dodge=True, ax=axes[0])
        axes[0].set_title('Stroke Disease by Work Type and Residence Type')
        axes[0].set_xlabel('Work Type')
        axes[0].set_ylabel('Stroke')
        axes[0].legend(title='Residence Type', loc='upper left')
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)

        # Stroke Disease by Smoking Status and Hypertension
        sns.barplot(data=df, x='smoking_status', y='stroke', hue='hypertension', palette=colors, ci=None, ax=axes[1])
        axes[1].set_title('Stroke Disease by Smoking Status and Hypertension')
        axes[1].set_xlabel('Smoking Status')
        axes[1].set_ylabel('Stroke')
        axes[1].legend(title='Hypertension')
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)

        st.pyplot(fig)
        st.write('Interpretasi: Hubungan antara jenis pekerjaan dan tipe tempat tinggal, status merokok dan hipertensi dengan penyakit stroke.')

# Run the Streamlit app
if __name__ == '__main__':
    app()
