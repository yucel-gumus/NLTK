import pandas as pd
import ast
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import pickle
import re

nltk.download('punkt')

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'genres', 'cast', 'overview', 'keywords', 'crew']]
movies.dropna(inplace=True)

def parse_list(obj):
    return [item['name'] for item in ast.literal_eval(obj)]

def parse_cast(obj):
    return [item['name'] for i, item in enumerate(ast.literal_eval(obj)) if i < 3]

def fetch_director(text):
    return [item['name'] for item in ast.literal_eval(text) if item['job'] == 'Director']

movies['genres'] = movies['genres'].apply(parse_list)
movies['keywords'] = movies['keywords'].apply(parse_list)
movies['cast'] = movies['cast'].apply(parse_cast)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

for column in ['genres', 'keywords', 'cast', 'crew']:
    movies[column] = movies[column].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

df = movies[['movie_id', 'title', 'tags']].copy()
df.loc[:, 'tags'] = df['tags'].apply(lambda x: " ".join(x).lower())

ps = PorterStemmer()
cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(df['tags'])
feature_names = cv.get_feature_names_out()

def simple_tokenize(text):
    return re.findall(r'\w+', text.lower())

stemmed_feature_names = [" ".join([ps.stem(word) for word in simple_tokenize(str(feature))]) for feature in feature_names]
unique_stemmed_feature_names = list(set(stemmed_feature_names))

similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_lower = movie.lower()
    movie_index = df[df['title'].str.lower() == movie_lower].index
    
    if not movie_index.empty:
        movie_index = movie_index[0]
        distances = similarity[movie_index]
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
        
        print(f"Movies similar to '{movie}':")
        for i, _ in movies_list:
            print(df.iloc[i].title)
    else:
        print(f"Movie '{movie}' not found in the database.")

recommend('Thor')

pickle.dump(df, open('movie_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))