#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split, cross_validate
import pandas as pd
import os

# Download MovieLens
data = Dataset.load_builtin('ml-100k')  # built-in MovieLens 100k

# Train/test split
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Use SVD (matrix factorization)
algo = SVD(n_factors=100, lr_all=0.005, reg_all=0.02, n_epochs=20)

# Train
algo.fit(trainset)

# Predict on testset
predictions = algo.test(testset)

# Evaluate
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)
print(f"Test RMSE: {rmse:.4f}, MAE: {mae:.4f}")

# Example: recommend top-N for a given user (user id as string in ML-100k)
def get_top_n(predictions, n=10):
    from collections import defaultdict
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    # sort predictions for each user and return top n
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

top_n = get_top_n(predictions, n=5)
# Show recommendations for an example user id '196' (as in MovieLens example)
print("Top 5 recommendations for user 196:", top_n.get('196', top_n.get(196)))

