import pandas as pd
import numpy as np
from string import punctuation
import re

from nltk.tokenize import regexp_tokenize, TweetTokenizer # funtions for tokenization
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import download
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import plot_confusion_matrix

import pickle

# Set random seed
SEED = 47

# Read Data

train = pd.read_csv('training_data/train.csv')
test = pd.read_csv('training_data/test.csv')
target = train['target']
test_ids = test['id'].copy()

# list of dataframes to preprocess

dframes = [train, test]

# functions for preprocessing

def drop_keyword_and_location_cols(df):
    df.drop(['keyword', 'location'], inplace=True, axis='columns')


def avg_word_length(element):
    word_lengths = [len(word) for word in element]
    return np.mean(word_lengths)


def tokenize_words(element):
    tt = TweetTokenizer()
    return tt.tokenize(element.lower())


def remove_punctuation_from_string(element):
    no_punc = element.translate(str.maketrans('', '', punctuation))
    no_punc = no_punc.lower()
    return no_punc


def punctuation_count(element):
    # Not including apostrophes
    counter = 0
    for letter in element:
        if letter in punctuation:
            counter += 1
    return counter


def extract_mentions(element):
    mentions_regex = r"@\w+"
    return regexp_tokenize(element, mentions_regex)


def includes_mention(element):
    if element == 0:
        return 0
    else:
        return 1

    
def extract_stopwords(element):
    try:
        stop_words = stopwords.words('english')
    except LookupError as le:
        download('stopwords')
        stop_words = stopwords.words('english')
    stop_words_present = []
    for word in element:
        if word in stop_words:
            stop_words_present.append(word)
    return stop_words_present


def remove_stopwords(element):
    try:
        stop_words = stopwords.words('english')
    except LookupError as le:
        download('stopwords')
        stop_words = stopwords.words('english')
        
    non_stopwords = []
    for word in element:
        if word not in stop_words:# and word != "'": # clean up stray apostrophes
            non_stopwords.append(word)
    return non_stopwords


def count_numbers(element):
    numbers_regex = r"(\d+\.?,?\s?\d+)"
    return len(regexp_tokenize(element, numbers_regex))


def extract_hashtags(element):
    hashtags_regex = r"#\w+"
    return regexp_tokenize(element, hashtags_regex)


def includes_hashtag(element):
    if element == 0:
        return 0
    else:
        return 1

    
def extract_weblinks(element):
    weblinks = []
    for token in tokenize_words(element):
        if token.startswith('http://'):
            weblinks.append(token)
    return weblinks


def includes_weblink(element):
    if element == 0:
        return 0
    else:
        return 1

    
def remove_links_and_stopwords(element):
    no_links_no_stopwords = []
    stop_words = stopwords.words('english')
    #for word in element.split(' '):
    for word in tokenize_words(element):
        if (word not in stop_words) & (not word.startswith('http')):
            no_links_no_stopwords.append(word)
    return no_links_no_stopwords


def lemmatizing(element):
    lem = WordNetLemmatizer()
    lemmed_words = [lem.lemmatize(word) for word in element]
    return lemmed_words


def remove_numbers(element):
    no_numbers = []
    for word in element:
        if not bool(re.search(r'\d', word)):
            no_numbers.append(word)
    return no_numbers


def list_to_text(element):
    return ' '.join(element)


def preprocess_text_cols(dframes):
    for df in dframes:
        drop_keyword_and_location_cols(df)
        df['lower_text'] = df['text'].str.lower()
        df['text_no_punc'] = df['text'].apply(remove_punctuation_from_string)
        df['tokens_no_punc'] = df['text_no_punc'].apply(tokenize_words)
        df['tokens_no_punc_no_stopwords'] = df['tokens_no_punc'].apply(remove_stopwords)
        df['tokens_no_punc_no_stopwords_no_links'] = df['text_no_punc'].apply(remove_links_and_stopwords)
        df['lemmed_words'] = df['tokens_no_punc_no_stopwords_no_links'].apply(lemmatizing)
        df['lemmed_words_no_numbers'] = df['lemmed_words'].apply(remove_numbers)
        df['final_text'] = df['lemmed_words_no_numbers'].apply(list_to_text)

# lowercase text
    # mentions_count
    # includes_mention
    # hashtags_count
    # includes_hashtag
    # punctuation_count
    # weblinks_count
    # includes weblink
# remove punctuation
    # number_count
# tokenize words
    # avg_word_length
    # num_words
    # remove stopwords
# remove stopwords
# remove links
# lemmatize words
# remove numbers
# transform to string
        
def create_new_features(dframes):
    for df in dframes:
        df['avg_word_length_feat'] = df['tokens_no_punc'].apply(avg_word_length)
        df['num_words_feat'] = df['tokens_no_punc'].apply(len)
        df['mentions'] = df['lower_text'].apply(extract_mentions)
        df['mention_count_feat'] = df['mentions'].apply(len)
        df['includes_mention_feat'] = df['mention_count_feat'].apply(includes_mention)
        df['stop_words'] = df['tokens_no_punc'].apply(extract_stopwords)
        df['stop_words_count_feat'] = df['stop_words'].apply(len)
        df['number_count_feat'] = df['text_no_punc'].apply(count_numbers)
        df['hashtags'] = df['lower_text'].apply(extract_hashtags)
        df['hashtag_count_feat'] = df['hashtags'].apply(len)
        df['includes_hashtag_feat'] = df['hashtag_count_feat'].apply(includes_hashtag)
        df['punctuation_count_feat'] = df['lower_text'].apply(punctuation_count)
        df['weblinks'] = df['lower_text'].apply(extract_weblinks)
        df['weblinks_count_feat'] = df['weblinks'].apply(len)
        df['includes_weblink_feat'] = df['weblinks_count_feat'].apply(includes_weblink)

preprocess_text_cols(dframes)
create_new_features(dframes)


# Create corpus for tf-idf vectorizer

def create_corpus(df):
    corpus = {}
    for index, row in df.iterrows():
        for word in row['lemmed_words_no_numbers']:
            if word in corpus:
                corpus[word] += 1
            else:
                corpus[word] = 1
    return corpus

word_dict = create_corpus(train)
sorted_word_dict = {k: v for k, v in sorted(word_dict.items(), key=lambda item: item[1], reverse=True)}

keys_to_remove = ['\x89', 'û', 'ûªs', '÷', 'ûª', '\x9d', 'ûò', 'ûªt', 'ûó' ]
for key in keys_to_remove:
    sorted_word_dict.pop(key)

corpus_dict = {k:v for (k,v) in sorted_word_dict.items() if v > 19}
corpus = [k for k,v in corpus_dict.items()]


tfidf_vect = TfidfVectorizer(vocabulary=corpus)
tfidf_vect_fit = tfidf_vect.fit(train['final_text'])
tfidf_train = tfidf_vect_fit.transform(train['final_text'])
tfidf_test = tfidf_vect_fit.transform(test['final_text'])
word_features_train_df = pd.DataFrame(tfidf_train.toarray())
word_features_test_df = pd.DataFrame(tfidf_test.toarray())


# Features List - only numerical features

features = [col for col in train.columns if col.endswith('feat')]


train = pd.concat([train[features].reset_index(drop=True), word_features_train_df, target.reset_index(drop=True)], axis=1)
test = pd.concat([test[features].reset_index(drop=True), word_features_test_df], axis=1)

# Features list

features = [col for col in train.columns if col not in ['target']]


# Build Ridge Classifier

ridge = RidgeClassifier()
params = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
          'max_iter': [1000]}
cross_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
gs = GridSearchCV(ridge, params, scoring='f1', cv=cross_val, verbose=1, return_train_score=True)
gs_fit = gs.fit(train[features], target)
results = pd.DataFrame(gs_fit.cv_results_).sort_values(by='mean_test_score', ascending=False)


ridge_winner = gs.best_estimator_
y_pred = ridge_winner.predict(test[features])

pickle.dump(ridge_winner, open('model.pkl', 'wb'))
pickle.dump(corpus, open('corpus.pkl', 'wb'))




