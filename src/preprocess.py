import nltk # type: ignore 
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize # type: ignore
from nltk.corpus import stopwords # type: ignore
import string
import numpy as np # type: ignore

def main():
    m = 8
    l =3
    file_path_positive = '../data/positive_tweets.txt'
    file_path_negative = '../data/negative_tweets.txt'

    pos_freqs = build_freq(file_path_positive)  # Build frequency dictionary
    # print("The psoitive frequency")
    # print(pos_freqs)
    neg_freqs = build_freq(file_path_negative)
    # print("The negative frquency")
    print(neg_freqs)
    word_f = word_freq(pos_freqs, neg_freqs, "../data/sample_sentiment_text.txt")
    print("The frequency of the sentences ares ", word_f)
    matrix = extract_features(word_f)
    print("The matrix is ", matrix)
    


def build_freq(file_path):
    freqs = {}
    # Build positive frequency paths
    # Process positive tweets
    process_file(file_path, freqs)
    # Process negative tweets
    #process_file(file_path_negative, freqs)
    return freqs

def process_file(file_path, freqs):
    
    with open(file_path, 'r', encoding='utf-8') as file:
        line = file.readline()
        while line:
            words = preprocess(line)
            for word in words:
                if word in freqs:
                    freqs[word] += 1
                else:
                    freqs[word] = 1
            line = file.readline()

def word_freq(pos_freq, neg_freq, file_path):
    wordsDic = {}
    count = 0
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            words = preprocess(line)
            word_dic = {}
            for word in words:
                lst = []
                if word in pos_freq:
                    lst.append(pos_freq[word])
                else:
                    lst.append(0)
                if word in neg_freq:
                    lst.append(neg_freq[word])
                else:
                    lst.append(0)
                word_dic[word] = lst

            wordsDic[count] = word_dic
            count += 1

    return wordsDic
def extract_features(words_dict):
    matrix = np.empty((0, 3))
    for count in words_dict:
        new = np.array([1, 0, 0])
        dics = words_dict[count]
        pos_tot = 0
        neg_tot = 0
        for word in dics:
            lst = dics[word]
            pos_tot += lst[0]
            neg_tot += lst[-1]
        new[1] = pos_tot
        new[2] = neg_tot
        matrix = np.vstack([matrix, new])
        
        
    return matrix


def preprocess(text):
    text = text.lower()
    
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens




if __name__ == "__main__":
    main()

#this files has the neccessary functions.
#that are distant from the model
#has function for creating word dictionary
#word frequency and more.