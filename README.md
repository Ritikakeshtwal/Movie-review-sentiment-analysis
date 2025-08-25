
# 🎬 Movie Review Sentiment Analysis (IMDB Dataset)

This project performs **Sentiment Analysis** on the IMDB movie reviews dataset using an **Artificial Neural Network (ANN)**.  
The model classifies each movie review as either **Positive** or **Negative**.  

## 📌 Features
- Uses the **IMDB Movie Review Dataset** (50,000 reviews)
- Text preprocessing with **sequence padding**
- ANN model with **Embedding + Dense + Dropout** layers
- Performance evaluation with:
  - Accuracy & Loss plots
  - Confusion Matrix heatmap
- Custom review prediction support

## ⚙️ Tech Stack
- Python  
- TensorFlow / Keras  
- NumPy  
- Scikit-learn  
- Matplotlib & Seaborn  

## 🚀 How to Run

# Clone the repository
git clone https://github.com/Ritikakeshtwal/Movie-Review-Sentiment-Analysis.
cd Movie-Review-Sentiment-Analysis

# Install dependencies
pip install -r requirements.txt

# Run the script
python imdb_sentiment.py


📊 Outputs
The following plots are saved in the outputs/ folder after training:

Accuracy Plot → outputs/accuracy.png

Loss Plot → outputs/loss.png

Confusion Matrix → outputs/confusion_matrix.png

✅ Example Prediction
yaml
Copy
Edit
Sample Review: the movie was absolutely wonderful fantastic performances and storyline
Prediction: Positive

📌 Future Enhancements
Use LSTM/GRU for sequence modeling

Hyperparameter tuning for better accuracy

Deploy as a web application using Flask/Streamlit



