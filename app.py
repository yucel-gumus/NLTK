import gradio as gr
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
            similar_movies.append(f"{similar_movie_title} (Benzerlik Skoru: {similarity_score:.2f})")

        return similar_movies

    else:
        return ["Benzer film bulunamadı."]

def gradio_interface(movie_title):
    df, similarity = load_data()
    return recommend_similar_movies(movie_title, df, similarity)

iface = gr.Interface(fn=gradio_interface, inputs="text", outputs="text", title="Film Önerici", description="Bir film adı girin ve benzer filmleri görün.")
iface.launch()
