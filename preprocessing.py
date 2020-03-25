import pandas as pd
import numpy as np
from string import punctuation
import re
import pickle

from nltk.tokenize import regexp_tokenize, TweetTokenizer # funtions for tokenization
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import download
nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer

def read_data(filepath):
	df = pd.read_csv(filepath)
	target = df['target']
	return df, target


def avg_word_length(element):
	word_lengths = [len(word) for word in element]
	return np.mean(word_lengths)


def tokenize_words(element):
	tt = TweetTokenizer()
	return tt.tokenize(element.lower())


def	remove_punctuation_from_string(element):
	no_punc = element.translate(str.maketrans('', '', punctuation))
	no_punc = no_punc.lower()
	return no_punc


def punctuation_count(element):
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


def preprocess_text_cols(df):
	df['lower_text'] = df['text'].str.lower()
	df['text_no_punc'] = df['text'].apply(remove_punctuation_from_string)
	df['tokens_no_punc'] = df['text_no_punc'].apply(tokenize_words)
	df['tokens_no_punc_no_stopwords'] = df['tokens_no_punc'].apply(remove_stopwords)
	df['tokens_no_punc_no_stopwords_no_links'] = df['text_no_punc'].apply(remove_links_and_stopwords)
	df['lemmed_words'] = df['tokens_no_punc_no_stopwords_no_links'].apply(lemmatizing)
	df['lemmed_words_no_numbers'] = df['lemmed_words'].apply(remove_numbers)
	df['final_text'] = df['lemmed_words_no_numbers'].apply(list_to_text)


def create_new_features(df):
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


def create_corpus(df):
	corpus = {}
	for index, row in df.iterrows():
		for word in row['lemmed_words_no_numbers']:
			if word in corpus:
				corpus[word] += 1
			else:
				corpus[word] = 1

	sorted_corpus_dict = {k: v for k, v in sorted(corpus.items(), key=lambda item: item[1], reverse=True)}
	keys_to_remove = ['\x89', 'û', 'ûªs', '÷', 'ûª', '\x9d', 'ûò', 'ûªt', 'ûó' ]

	for key in keys_to_remove:
		sorted_corpus_dict.pop(key)

	corpus_dict = {k:v for (k,v) in sorted_corpus_dict.items() if v > 19}
	corpus = [k for k,v in corpus_dict.items()]

	return corpus


def tf_idf(df, col, vocabulary):
	tfidf_vect = TfidfVectorizer(vocabulary=vocabulary)
	tfidf_vect_fit = tfidf_vect.fit(df[col])
	tfidf_train = tfidf_vect_fit.transform(df[col])
	word_features_train_df = pd.DataFrame(tfidf_train.toarray())
	return word_features_train_df


def return_final_df(df1, df2, target_series=None):
	numeric_features = [col for col in df1.columns if col.endswith('feat')]
	train = pd.concat([df1[numeric_features].reset_index(drop=True), df2], axis=1)
	features = [col for col in train.columns if col not in ['target']]
	if target_series is None:
		return train, features
	else:
		train = pd.concat([train, target_series.reset_index(drop=True)], axis=1)
	return train, features


def main():
	train_filepath = 'training_data/train.csv'
	train, target = read_data(train_filepath)
	preprocess_text_cols(train)
	create_new_features(train)
	corpus = create_corpus(train)
	pickle.dump(corpus, open('corpus.pkl', 'wb'))
	word_features_train_df = tf_idf(train, 'final_text', corpus)
	train, features = return_final_df(train, word_features_train_df, target)


if __name__ == '__main__':
	main()

