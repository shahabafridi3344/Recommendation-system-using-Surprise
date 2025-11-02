# AI Project 4: Movie Recommendation System (Collaborative Filtering)

## Overview
This project implements a simple movie recommendation system using **Collaborative Filtering**. 
It recommends movies to users based on the similarity of their ratings with other users.

## Features
- User-based collaborative filtering using cosine similarity
- Movie recommendations based on user–item matrix
- Built using pandas, numpy, and scikit-learn
- Easily extendable for larger datasets like MovieLens

## Installation
Run this cell in your Jupyter notebook to install required libraries:
```python
import sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas", "numpy", "scikit-learn"])
```

## Usage
1. Create or load a dataset with User, Movie, and Rating columns.
2. Generate a user–movie rating matrix.
3. Compute similarity between users using cosine similarity.
4. Get movie recommendations for a specific user based on similar users’ preferences.

## Example Output
```
🎬 Sample Ratings Data:
   User      Movie  Rating
0     A  Inception       5
1     A   Avengers       4
2     A    Titanic       3
3     B  Inception       5
4     B    Titanic       4
...

🎥 User-Movie Matrix:
 Movie  Avengers  Avatar  Inception  Titanic
User
A           4.0     NaN        5.0      3.0
B           NaN     NaN        5.0      4.0
C           4.0     5.0        NaN      5.0
D           NaN     4.0        4.0      NaN

👯 User Similarity Matrix:
        A      B      C      D
A  1.000  0.981  0.899  0.956
B  0.981  1.000  0.921  0.962
C  0.899  0.921  1.000  0.932
D  0.956  0.962  0.932  1.000

🎯 Recommended movies for User A: ['Avatar']
```

## Files
- `recommendation_system.ipynb`: Jupyter notebook with full code
- `README_Recommendation_System.txt`: Documentation
