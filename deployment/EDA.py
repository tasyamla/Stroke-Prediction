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
        st.write('There were 4.87% of patients suffering from stroke and 95.13% of patients did not suffer from stroke. In general, the proportion of stroke sufferers is relatively small in the dataset.')

        # Distribution of Patient Ages
        st.subheader('Distribution of Patient Ages')
        fig, ax = plt.subplots()
        sns.histplot(df['age'], kde=True, stat='density', discrete=True, color='coral', alpha=0.6, ax=ax)
        ax.set_title('Age Distribution')
        ax.grid(False)
        st.pyplot(fig)
        st.write('Based on the visualization of the age distribution, it is known that the age distribution of patients is 0 - 82 years with the highest age being 78 years. This shows that the importance of the age factor in stroke disease, the risk of suffering a stroke tends to increase with age.')

        # Distribution of Blood Glucose Levels
        st.subheader('Distribution of Blood Glucose Levels')
        fig, ax = plt.subplots()
        sns.histplot(df['avg_glucose_level'], kde=True, stat='density', discrete=True, color='skyblue', alpha=0.6, ax=ax)
        ax.set_title('Distribution of Glucose Levels')
        ax.grid(False)
        st.pyplot(fig)
        st.write('The patients blood glucose levels ranged from 55.12 - 271.74 mg/dL. The average patient had a blood glucose level of 106.14 mg/dL. The majority of patients have normal glucose levels (70 - 110 mg/dL) which can be seen on a high curve in the normal range. Although the majority of patients have normal glucose levels, the distribution also includes patients with abnormal glucose levels. This indicates variations in blood glucose levels among patients.')

        # Distribution of BMI
        st.subheader('Distribution of Body Mass Index')
        fig, ax = plt.subplots()
        sns.histplot(df['bmi'], kde=True, stat='density', discrete=True, color='skyblue', alpha=0.6, ax=ax)
        ax.set_title('Distribution of Body Mass Index')
        ax.grid(False)
        st.pyplot(fig)
        st.write('Based on visualization of the BMI distribution, it is known that the patients BMI is in the range 10.3 - 97.6 Kg/m².')

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
        Based on visualization of stroke risk factors, also known:
        - The majority of patients are female at 58.60% compared to men at 41.40%.
        - Around 90.25% of patients have no history of hypertension and 9.75% of patients have a history of hypertension.
        - Around 94.60% of patients do not have heart disease and 5.40% of patients have heart disease.
        - The majority of patients are married, 65.63%, while those who are not married are 34.37%.
        - The majority of patients live in urban areas, 50.81% compared to 49.19% in rural areas.
        - Approximately 37% of patients do not smoke, 30.22% of patients' smoking status is unknown, 17.30% of patients have previously smoked, and 15.44% of patients smoke.
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
        interpretation = """
        Based on the visualization of work types, also known the majority of patients work in private companies, 2924 patients (57.23%), then 819 patients (16.03%) work self-employed, 687 patients (13.45%) are still children, 657 patients (12.86%) ) worked in Gov Jobs and 22 patients (0.43%) had never worked.
        """
        st.write(interpretation)

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
        interpretation = """
        **Explanation:**

        1. **Gender**:
            - The majority of male and female patients do not suffer from stroke

        2. **Hypertension**:
            - The majority of patients who do not suffer a stroke also do not have hypertension.
            - There was a slight increase in patients who had hypertension and suffered stroke, suggesting that hypertension may be a risk factor for stroke.

        3. **Heart Disease**:
            - The majority of patients who do not suffer a stroke also do not have heart disease.
            - There was a slight increase in patients who had heart disease and suffered a stroke, suggesting that heart disease may be a risk factor for stroke.

        4. **Ever Married**:
            - The majority of patients who have ever been married suffer from stroke.
            - The distribution of patients who suffer stroke is more in patients who have been married, but the difference is not very significant.

        5. **Residence Type**:
            - Both in urban and rural areas, the number of stroke cases is lower compared to those who have not had a stroke.
            - The distribution between urban and rural areas appears to be similar, with almost the same proportion of stroke and non-stroke cases.

        6. **Smoking Status**:
            - The majority of patients who do not suffer a stroke are non-smokers.
            - There was a slight increase in patients who smoked and suffered stroke, suggesting that smoking may be a risk factor for stroke.

        7. **Work Type**:
            - The majority of patients who do not suffer a stroke work in the private sector.
            - The number of stroke cases is lower in all types of work compared to those who did not experience a stroke, but the difference is not very significant.
        """
        st.write(interpretation)
    

        # Relationship between Numerical Features and Stroke
        st.subheader('Relationship between Numerical Features and Stroke')
        features = df[['age', 'avg_glucose_level', 'bmi', 'stroke']]
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(features.corr(), cmap="vlag", annot=True, linewidths=.8, ax=ax)
        st.pyplot(fig)
        interpretation = """
        **Explanation:**
        Based on the visualization of data regarding the relationship between various variables, also known that:

        - `Age and Stroke`: There was a correlation between the age and stroke variables of 0.25, indicating a positive relationship. The older a person gets, the more likely they are to have a stroke, although this relationship is moderately correlated. 

        - `Average Glucose Levels and Stroke`: There is a correlation between the glucose level variable and stroke of 0.13, which indicates a positive but quite weak relationship. The higher the blood glucose level, the more likely it is to have a stroke. 

        - `BMI and Stroke`: There was no correlation between the BMI and stroke variables of 0.042.

        """
        st.write(interpretation)

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
        interpretation = """
        **Explanation:**
        
        **Stroke by Marital Status and Gender**

        - Men and women who have been married have a higher proportion of stroke incidence compared to those who have never been married.

        - Men who have been married have a higher percentage of stroke incidence than women who have been married.

        - Men have a higher risk of stroke compared to women. The difference in stroke risk in men or women may be influenced by lifestyle differences, such as smoking habits and higher alcohol consumption in men. In addition, marital status can increase the risk of stroke, especially in men, it can occur due to factors such as stress and family responsibilities.

        **Stroke based on Hypertension and Heart Disease**

        - Patients who have a history of hypertension and also have heart disease are at a very high risk of having a stroke.

        - The association between hypertension and heart disease shows the highest risk for stroke events. This shows that prevention and management efforts for both health conditions are essential to reduce the risk of stroke.

        - Hypertension causes narrowing or rupture of blood vessels in the brain which will result in impaired blood flow to the brain and even cause the death of brain cells. 
        """
        st.write(interpretation)

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
        interpretation = """
        **Explanation:**
        
        **Stroke by Type of Work and Residence Type**

        The incidence of stroke in patients who have a business (entrepreneur) and live in urban areas is higher than those who live in rural areas.  This suggests that employers may have certain risk factors that increase the incidence of stroke, such as higher stress. In addition, patients who live in cities have a more sedentary lifestyle or higher levels of stress in urban work environments.
        """
        st.write(interpretation)

# Run the Streamlit app
if __name__ == '__main__':
    app()
