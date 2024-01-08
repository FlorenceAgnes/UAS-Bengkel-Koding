import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings("ignore")

st.set_option('deprecation.showPyplotGlobalUse', False)

# Inisialisasi StandardScaler
sc = StandardScaler()

# Inisialisasi variable classifier
classifier = None

# Fungsi untuk load dan preprocess dataset
def load_and_preprocess_data():
    global classifier, X_train, X_test, y_train, y_test

    file_name = 'processed.cleveland.data'
    column_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(file_name, header=None, names=column_names)
    df['target'] = df['target'].replace([2, 3, 4], 1)
    df['sex'] = pd.to_numeric(df['sex'], errors='coerce').astype('float')
    df['ca'] = pd.to_numeric(df['ca'], errors='coerce').astype('float')
    df['thal'] = pd.to_numeric(df['thal'], errors='coerce').astype('float')
    df = df.dropna(subset=['ca', 'thal'])

    # Fit StandardScaler pada data
    df_scaled = pd.DataFrame(sc.fit_transform(df.iloc[:, :-1]), columns=df.columns[:-1])

    # Extract variabel features dan target
    X = df_scaled.values
    y = df['target'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    classifier = GaussianNB()

    # Fit classifier
    classifier.fit(X_train, y_train)

    return df
    

# Fungsi untuk klasifikasi dengan Naive Bayes
def naive_bayes_classification(df):
    df['thal'] = df.thal.fillna(df.thal.mean())
    df['ca'] = df.ca.fillna(df.ca.mean())

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Klasifikasi dengan Naive Bayes
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    # Prediksi pada test set
    y_pred = classifier.predict(X_test)

    # Confusion Matrix
    cm_test = confusion_matrix(y_pred, y_test)
    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    # Akurasi untuk training dan test set
    accuracy_train = (cm_train[0][0] + cm_train[1][1]) / len(y_train)
    accuracy_test = (cm_test[0][0] + cm_test[1][1]) / len(y_test)

    # Tampilkan hasil
    st.write("### Hasil Klasifikasi Naive Bayes:")
    st.write(f"Akurasi untuk training set: {accuracy_train:.4f}")
    st.write(f"Akurasi untuk test set: {accuracy_test:.4f}")
    
    # Confusion Matrix - Train Set
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
    plt.title('Confusion Matrix - Train Set')
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    st.pyplot()
    
    # Confusion Matrix - Test Set
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    st.pyplot()

# Main Streamlit
def main():
    st.title("Heart Disease Dataset Exploration")

    st.sidebar.subheader("Navigation")
    page = st.sidebar.radio("Go to", ["Eksplorasi Data", "Visualisasi Data", "Klasifikasi Naive Bayes", "Klasifikasi Decision Tree", "Klasifikasi Logistic Regression", "Prediksi"])

    if page == "Eksplorasi Data":
        st.sidebar.subheader("Eksplorasi Data")
        st.write("### Dataset Preview:")
        df = load_and_preprocess_data()
        st.dataframe(df.head())

        st.write("### Informasi Dataset:")
        st.write(df.info())

        st.write("### Descriptive Statistics:")
        st.write(df.describe())

        st.write("### Unique Values:")
        display_unique_values(df)

        st.write("### Missing Values:")
        missing_values = df.isnull().sum()
        st.write(missing_values)

    elif page == "Visualisasi Data":
        st.sidebar.subheader("Visualisasi Data")
        visualize_data(load_and_preprocess_data())
        
    elif page == "Klasifikasi Naive Bayes":
        st.sidebar.subheader("Klasifikasi Naive Bayes")
        naive_bayes_classification(load_and_preprocess_data())

    elif page == "Klasifikasi Decision Tree":
        st.sidebar.subheader("Klasifikasi Decision Tree")
        decision_tree_classification(load_and_preprocess_data())

    elif page == "Klasifikasi Logistic Regression":
        st.sidebar.subheader("Klasifikasi Logistic Regression")
        logistic_regression_classification(load_and_preprocess_data())

    elif page == "Prediksi":
        st.sidebar.subheader("Prediksi")
        classifier_choice = st.radio("Pilih Classifier", ["Naive Bayes", "Decision Tree", "Logistic Regression"])
        predict_page(load_and_preprocess_data(), classifier_choice)

# Function untuk menampilkan unique values setiap kolom
def display_unique_values(df):
    for column in df.columns:
        unique_values = df[column].unique()
        st.write(f"Unique values pada '{column}':\n{unique_values}\n")

# Visualisasi Data
def visualize_data(df):
    # Visualisasi 1: Distribusi Target berdasarkan Jenis Kelamin
    st.write("#### Distribusi Target berdasarkan Jenis Kelamin:")
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.countplot(x='sex', hue='target', data=df, palette='pastel')
    plt.title('Distribusi Target berdasarkan Jenis Kelamin')
    plt.xlabel('Jenis Kelamin (0 = wanita, 1 = pria)')
    plt.ylabel('Jumlah')
    st.pyplot()

    # Visualization 2: Histogram Distribusi Umur berdasarkan Target
    st.write("#### Distribusi Umur berdasarkan Target:")
    plt.figure(figsize=(12, 8))
    plt.hist(df[df['target'] == 0]['age'], bins=20, alpha=0.5, label='No Heart Disease', color='blue')
    plt.hist(df[df['target'] == 1]['age'], bins=20, alpha=0.5, label='Heart Disease', color='orange')
    plt.title('Distribusi Umur berdasarkan Target')
    plt.xlabel('Umur')
    plt.ylabel('Jumlah')
    plt.legend()
    st.pyplot()

# Fungsi untuk klasifikasi dengan Decision Tree
def decision_tree_classification(df):
    df['thal'] = df.thal.fillna(df.thal.mean())
    df['ca'] = df.ca.fillna(df.ca.mean())

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Klasifikasi Decision Tree
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train, y_train)

    # Prediksi pada test set
    y_pred = classifier.predict(X_test)

    # Confusion Matrix
    cm_test = confusion_matrix(y_pred, y_test)
    
    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    # Accuracy untuk training and test set
    accuracy_train = (cm_train[0][0] + cm_train[1][1]) / len(y_train)
    accuracy_test = (cm_test[0][0] + cm_test[1][1]) / len(y_test)

    # Tampilkan hasil
    st.write("### Hasil klasifikasi dengan Decision Tree Classification:")
    st.write(f"Akurasi untuk training set: {accuracy_train:.4f}")
    st.write(f"Akurasi untuk test set: {accuracy_test:.4f}")
    
    # Confusion Matrix - Train Set
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
    plt.title('Confusion Matrix - Train Set')
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    st.pyplot()

    # Confusion Matrix - Test Set
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    st.pyplot()

# Fungsi untuk klasifikasi dengan Logistic Regression
def logistic_regression_classification(df):
    df['thal'] = df.thal.fillna(df.thal.mean())
    df['ca'] = df.ca.fillna(df.ca.mean())

    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Feature scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Klasifikasi Logistic Regression
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    # Predictions test set
    y_pred = classifier.predict(X_test)

    # Confusion Matrix
    cm_test = confusion_matrix(y_pred, y_test)
    y_pred_train = classifier.predict(X_train)
    cm_train = confusion_matrix(y_pred_train, y_train)

    # Akurasi for training and test set
    accuracy_train = (cm_train[0][0] + cm_train[1][1]) / len(y_train)
    accuracy_test = (cm_test[0][0] + cm_test[1][1]) / len(y_test)

    # Tampilkan hasil
    st.write("### Hasil Klasifikasi Logistic Regression:")
    st.write(f"Akurasi untuk training set: {accuracy_train:.4f}")
    st.write(f"Akurasi untuk test set: {accuracy_test:.4f}")
    
    # Confusion Matrix - Train Set
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
    plt.title('Confusion Matrix - Train Set')
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    st.pyplot()
    
    # Confusion Matrix - Test Set
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'])
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')
    st.pyplot()

# Fungsi Prediksi
def predict_page(df, classifier_choice):
    global classifier, X_train, y_train
    st.write("### Input User untuk Prediksi:")
    
    user_input = {}

    # Input untuk sex (0 or 1)
    user_input['sex'] = st.radio("Select sex", [0, 1])

    # Input untuk fbs (0 or 1)
    user_input['fbs'] = st.radio("Select fbs", [0, 1])

    # Input untuk exang (0 or 1)
    user_input['exang'] = st.radio("Select exang", [0, 1])

    # Input untuk cp (1, 2, 3, 4)
    user_input['cp'] = st.selectbox("Select cp", [1, 2, 3, 4])

    # Input untuk restecg (0, 1, 2)
    user_input['restecg'] = st.selectbox("Select restecg", [0, 1, 2])

    # Input untuk slope (1, 2, 3)
    user_input['slope'] = st.selectbox("Select slope", [1, 2, 3])

    # Input untuk ca (0, 1, 2, 3)
    user_input['ca'] = st.selectbox("Select ca", [0, 1, 2, 3])

    # Input untuk thal (3, 6, 7)
    user_input['thal'] = st.selectbox("Select thal", [3, 6, 7])

    #  Set slider
    for column in df.columns[:-1]:
        if column not in ['sex', 'fbs', 'exang', 'cp', 'restecg', 'slope', 'ca', 'thal']:
            user_input[column] = st.slider(f"Enter {column}", df[column].min(), df[column].max(), df[column].mean())

    # Feature scaling for user input
    user_input_scaled = sc.transform(pd.DataFrame([list(user_input.values())], columns=df.columns[:-1]).values)

    # Pilih classifier
    if classifier_choice == "Naive Bayes":
        # Fit model Naive Bayes 
        classifier = GaussianNB()
    elif classifier_choice == "Decision Tree":
        # Fit model Decision Tree
        classifier = DecisionTreeClassifier()
    elif classifier_choice == "Logistic Regression":
        # Fit model Logistic Regression
        classifier = LogisticRegression()

    # Fit classifier yang dipilih
    classifier.fit(X_train, y_train)

    # Prediksi untuk input user
    user_prediction = classifier.predict(user_input_scaled)[0]

    # Tampilkan hasil
    st.write(f"### Hasil Prediksi:")
    st.write(f"Prediksi Model: {user_prediction} (0: No Heart Disease, 1: Heart Disease)")

if __name__ == "__main__":
    main()