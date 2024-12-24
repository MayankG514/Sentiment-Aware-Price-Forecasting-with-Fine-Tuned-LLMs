Requirements to Run the Code:
Python 3.x (Recommended version: 3.7 or higher)
Jupyter Lab (To run the notebooks)
Python Libraries : 
    NLTK
    scikit-learn
    matplotlib
    torch
    transformers
    tqdm

Dataset:
Download the dataset from the link below:

Cryptocurrency News Dataset: https://drive.google.com/drive/folders/1ijF7fg5QeakQNBZDenkBGs8gGlR3-4Hp?usp=drive_link

FinBert Model:
We will use the FinBert model for sentiment analysis: https://huggingface.co/models?other=base_model:finetune:ProsusAI/finbert


Problem Description:
Cryptocurrency markets are growing rapidly, generating massive amounts of real-time data. Investors are faced with the challenge of making 
timely decisions (buy, sell, or hold) in such a fast-moving market. This project proposes the use of a fine-tuned Large Language Model (LLM) 
to classify the sentiment of cryptocurrency news headlines and help investors make informed decisions.

Solution:
The project fine-tunes a FinBert model to classify the sentiment of Bitcoin and Ethereum news headlines. We also explore the integration of 
sentiment analysis with price prediction, where the model provides both sentiment labels and price predictions.

Steps Performed in the Code:
1. Data Preprocessing: Load and clean data from the provided JSON files.
2. Merge the Bitcoin and Ethereum datasets, tokenize news headlines, and pad sentences to the required input size.
3. Model Preparation: Fine-tune the FinBert model for sentiment classification and adapt a regression model for price prediction.
4. Training: Train the model using a multi-task learning approach with sentiment classification and price prediction.
5. Evaluation: Evaluate the model performance using a confusion matrix and accuracy scores.
6. Model Comparison: Compare the models trained with different parameters and choose the best one.

How to Run:
Install required dependencies
Launch Jupyter Lab and run the following notebooks:

Sentiment analysis of news: SentimentAnalysis.ipynb
Sentiment Analysis and price prediction with news as input: SentimentAnalysis_and_PricePrediction.ipynb