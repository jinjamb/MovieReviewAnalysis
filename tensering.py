from transformers import BertTokenizer, BertModel
import torch
import os

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

tokenizedreviewfolders = ["..\\SentimentAnalysis\\tokenizedreviewpos", "..\\SentimentAnalysis\\tokenizedreviewneg"]

for elements in tokenizedreviewfolders:

tokens = tokenizer(review, padding="max_length", truncation=True, max_length=10, return_tensors='pt')
print(tokens)