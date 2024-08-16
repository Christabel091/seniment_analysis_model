#test data
from preprocess import word_freq, extract_features, build_freq
import numpy as np
decision_boundary = 0.5
#Now the model has been trained, now it is time to test the model and see how accyrate it is. 
parameter = [ 0.005, 0.07612883, -0.06773525]
b= [-0.04217752]
def model(matrix, parameter, b):
    y_hat = []
    for row in matrix:
        feature = row
        z = np.dot(feature, parameter) + b
        g =  1 / (1 + np.exp(-z))
        y_hat.append(g)
    return np.array(y_hat)
file_path_positive = '../data/positive_tweets.txt'
file_path_negative = '../data/negative_tweets.txt'
import math
pos_freqs = build_freq(file_path_positive)  # Build frequency dictionary
neg_freqs = build_freq(file_path_negative)
file_name = input("What is the name of the file you want to find the sentiment of the tweets? ")
file_name = "../output/" + file_name
word_f = word_freq(pos_freqs, neg_freqs, file_name)
matrix = extract_features(word_f)
y_hat = model(matrix, parameter, b)
decision_boundary = 0.5
print("These are the sentiment of the text that are in your file.")
print("These sentiment are also written to a file 'predicted.txt'.")
with open("../output/predicted.txt", "w") as file:
    for row in y_hat:
        if row[0] > decision_boundary:
            file.write("tweet is a nice tweet" + "\n")
            print("tweet is a nice tweet")
        else:
            print("Tweet is a negative tweet.")
            file.write("tweet is a negative tweet"+ "\n")