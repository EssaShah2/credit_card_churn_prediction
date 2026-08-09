# 💳 Credit Card Churn Prediction

[![Streamlit App](https://img.shields.io/badge/🚀_Live_App-Click_Here-blue?style=for-the-badge&logo=streamlit)](https://creditcardchurnprediction-fn.streamlit.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/EssaShah2/credit_card_churn_prediction)

## 📌 Project Overview

This project provides an end-to-end machine learning solution for predicting credit card customer churn. It uses a deep learning model built with **TensorFlow/Keras** to identify customers likely to leave, enabling data-driven retention strategies.

## ✨ Key Features

- **Deep Learning Model**: Built with TensorFlow/Keras, optimized for tabular data.
- **Comprehensive Preprocessing**: Uses `scikit-learn` pipelines for scaling and encoding.
- **Interactive Web App**: Deployed with Streamlit for real-time, user-friendly predictions.
- **Production-Ready Artifacts**: Exports the trained model, scaler, and transformer for seamless deployment.

## 🔗 Quick Links

- **Live Application**: [creditcardchurnprediction-fn.streamlit.app](https://creditcardchurnprediction-fn.streamlit.app/)
- **Source Code**: [GitHub Repository](https://github.com/EssaShah2/credit_card_churn_prediction)

## 🛠️ Tech Stack

- **Language**: Python 3.11+
- **Machine Learning**: TensorFlow, Keras, scikit-learn
- **Data Processing**: Pandas, NumPy
- **Web Framework**: Streamlit
- **Serialization**: Keras (.keras), Pickle (.pkl)

## 📁 Repository Structure

```
credit_card_churn_prediction/
├── app.py                  # Streamlit web application
├── churn_model.keras       # Trained TensorFlow/Keras model
├── scaler.pkl              # Fitted StandardScaler for feature scaling
├── transformer.pkl         # Fitted ColumnTransformer for encoding
├── requirements.txt        # Python package dependencies
├── .gitignore              # Files and directories to ignore in Git
├── .python-version         # Python version specification
└── README.md               # Project documentation (this file)
```

## 🚀 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/EssaShah2/credit_card_churn_prediction.git
    cd credit_card_churn_prediction
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

    The application will open in your default web browser.

## 👨‍💻 Author

**Muhammad Essa Shah**
- GitHub: [EssaShah2](https://github.com/EssaShah2)
- LinkedIn: [essa-shah22](https://www.linkedin.com/in/essa-shah22/)
- Kaggle: [ronaldoshortsvideo](https://www.kaggle.com/ronaldoshortsvideo)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
