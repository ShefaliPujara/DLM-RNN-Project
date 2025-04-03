
# Sentiment Analysis on COVID-19 Tweets Using RNN

## By: Shefali Pujara

### Tools & Libraries
Keras, TensorFlow, Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, NLTK

## 1. Introduction
This project focuses on performing sentiment analysis on tweets related to COVID-19 using Recurrent Neural Networks (RNN). The aim is to classify tweets as **Extremely Positive, Positive, Neutral, Negative, or Extremely Negative** based on public sentiment. The model is fine-tuned with hyperparameter optimization and evaluated on real-world COVID-19 tweet data.

## 2. Dataset Description

**COVID-19 Tweets Dataset**
- **Training Set:** 40,000 samples
- **Test Set:** 10,000 samples

**Labels:**
- Extremely Positive
- Positive
- Neutral
- Negative
- Extremely Negative

## 3. Objective
The primary objectives of this project are:
- To build an **RNN-based model** for sentiment classification of COVID-19 tweets.
- To perform **text preprocessing** to clean and structure the dataset.
- To **fine-tune hyperparameters** to improve model performance.
- To evaluate the model using various **performance metrics** and visualization techniques.

## 4. Data Preprocessing

### 4.1 Data Cleaning
The dataset undergoes multiple preprocessing steps:
- **Lowercasing:** Ensuring uniform text format.
- **Removing URLs, mentions (@usernames), and hashtags** to eliminate noise.
- **Removing special characters, numbers, and punctuation** for better tokenization.
- **Removing stopwords** to focus on meaningful words.
- **Tokenization:** Converting text into numerical sequences.
- **Padding sequences** to ensure consistent input size for the model.

### 4.2 Encoding Labels
- Mapping **Extremely Positive, Positive, Neutral, Negative, and Extremely Negative** labels into numerical categories for multi-class classification.

## 5. Model Architecture and Iterations

### 5.1 Baseline Model
We started with a **basic RNN model** as an initial benchmark.

- **Embedding Layer:** 100 dimensions
- **RNN Layer:** 128 units
- **Dense Layers:**
  - 128 units with ReLU activation
  - Output layer with Softmax activation (for multi-class classification)
- **Dropout:** 0.2 (to reduce overfitting)
- **Optimizer:** Adam with a learning rate of 0.0001
- **Loss Function:** Categorical Crossentropy
- **Callbacks:**
  - EarlyStopping (patience = 5)
  - ModelCheckpoint to save the best model

**Results:**
- Validation Accuracy: **78.6%**
- Test Accuracy: **77.9%**
- Neutral sentiment was **harder to classify**, with some misclassifications between positive and neutral.

### 5.2 Enhanced Model with Hyperparameter Tuning
To improve performance, the following changes were made:
- **Increased Embedding Dimension**: 128
- **Adjusted Dropout Rate**: 0.3
- **Fine-tuned Learning Rate**: 0.00005
- **Batch Size**: 64
- **Epochs**: 25

**Results:**
- Validation Accuracy: **80.2%**
- Test Accuracy: **79.4%**
- Slightly better generalization but some overfitting noticed.

### 5.3 Bidirectional RNN for Context Capture
To capture context better in both directions, we implemented a **Bidirectional RNN**.

- **Bidirectional RNN Layer:** 128 units
- **Batch Normalization** for better convergence
- **Gradient Clipping** to prevent exploding gradients

**Results:**
- Validation Accuracy: **81.7%**
- Test Accuracy: **80.5%**
- Misclassification of **neutral tweets** remains a challenge.

## 6. Model Evaluation and Performance Analysis

### 6.1 COVID-19 Tweet Dataset Performance
- The RNN model achieved around **80% accuracy** on the test dataset.
- **Confusion Matrix:**
  - Extremely Positive and Extremely Negative sentiments were classified well.
  - Neutral sentiment had a higher misclassification rate.
- **Loss and Accuracy Trends:**
  - Loss decreased steadily, indicating effective learning.

### 6.2 Limitations and Generalization Issues
- The model showed **bias towards positive sentiments**, classifying neutral tweets incorrectly.
- **Short tweets** lacked enough context, making classification harder.

## 7. Visualizations

### 7.1 Accuracy and Loss Curves
- **Training and validation accuracy** improved steadily over epochs.
- **Loss curves** showed proper convergence, ensuring stable learning.

### 7.2 Confusion Matrix Insights
- Many **neutral tweets** were misclassified as positive.
- **Negative tweets** were more accurately predicted compared to neutral ones.

## 8. Conclusion
In this project:
- We successfully built an **RNN-based sentiment analysis model** for COVID-19 tweets.
- Applied **hyperparameter tuning** and **regularization** to improve accuracy.
- Achieved **80.5% accuracy**, but struggled with **neutral sentiment classification**.

## 9. Observations and Managerial Insights

### Observations:
- **Positive Sentiment (27.75%)** is the most common in the dataset.
- **Negative Sentiment (24.09%)** is also significant, showing public concerns.
- **Neutral Tweets (18.74%)** indicate a substantial portion of tweets are factual rather than opinionated.
- **Extremely Positive (16.09%)** and **Extremely Negative (13.31%)** sentiments highlight strong emotional responses.
- **Longer tweets** contain more context, leading to better predictions, while **short tweets** often get misclassified.

### Managerial Insights:
- **Public Perception Analysis:**
  - Businesses can use sentiment analysis to gauge **customer concerns and opinions** on health-related topics.
  - Governments can monitor **public sentiment trends** to adjust communication strategies.
- **Crisis Management:**
  - Identifying shifts in sentiment allows organizations to **proactively address misinformation**.
  - Policy adjustments can be made based on **real-time feedback from social media**.
- **Marketing Strategies:**
  - Brands can **align messaging** with prevailing sentiments to engage audiences effectively.
  - Companies can detect **negative sentiment spikes** and respond swiftly to prevent reputation damage.
- **Technology Enhancements:**
  - Using **advanced models like LSTM, GRU, or Transformer-based architectures** can enhance accuracy.
  - **Training on more diverse datasets** will improve model generalization to real-world sentiments.

This project highlighted the strengths and challenges of **RNNs for NLP tasks**, showing the need for **advanced architectures** like LSTM, GRU, or Transformers for better accuracy and generalization.

