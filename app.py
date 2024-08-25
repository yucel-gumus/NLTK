from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def load_data():
    try:
        df = pickle.load(open('movie_list.pkl', 'rb'))
        similarity = pickle.load(open('similarity.pkl', 'rb'))
    except FileNotFoundError:
        print("Pickle dosyaları bulunamadı. Veri setini yeniden işliyoruz...")
        df, similarity = process_data()
    
    df['title_lower'] = df['title'].str.lower()
    return df, similarity

def process_data():

    movies = pd.read_csv('tmdb_5000_movies.csv')
    movies = movies[['title', 'overview']]
    movies['tags'] = movies['overview']
    
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies['tags'].fillna(''))
    similarity = cosine_similarity(vectors)
    
    pickle.dump(movies, open('movie_list.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))
    
    return movies, similarity

def recommend_similar_movies(movie_title, df, similarity):
    movie_title_lower = movie_title.lower()

    movie_index = df[df['title_lower'] == movie_title_lower].index

    if not movie_index.empty:
        movie_index = movie_index[0]

        distances = similarity[movie_index]

        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        similar_movies = []
        for i in movies_list:
            similar_movie_title = df.iloc[i[0]].title
            similarity_score = i[1]
            similar_movies.append({"title": similar_movie_title, "score": similarity_score})

        return similar_movies

    else:
        return []  

@app.route('/', methods=['GET', 'POST'])
def index():
    movie_title = ""
    similar_movies = []
    error_message = ""

    if request.method == 'POST':
        user_input = request.form['movie_title']

        df, similarity = load_data()

        similar_movies = recommend_similar_movies(user_input, df, similarity)
        movie_title = user_input

        if not similar_movies:
            error_message = f"'{movie_title}' filmi bulunamadı veya benzer film önerisi yok."

    return render_template('index.html', movie_title=movie_title, similar_movies=similar_movies, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)