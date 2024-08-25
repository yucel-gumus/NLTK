import pickle

def load_data():
    df = pickle.load(open('movie_list.pkl', 'rb'))
    similarity = pickle.load(open('similarity.pkl', 'rb'))

    df['title_lower'] = df['title'].str.lower()

    return df, similarity

def recommend_similar_movies(movie_title, df, similarity):
    movie_title_lower = movie_title.lower()

    movie_index = df[df['title_lower'] == movie_title_lower].index

    if not movie_index.empty:
        movie_index = movie_index[0]

        distances = similarity[movie_index]

        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        print(f"Similar movies to {df.iloc[movie_index].title}:")
        for i in movies_list:
            similar_movie_title = df.iloc[i[0]].title
            similarity_score = i[1]
            print(f"{similar_movie_title} - Similarity Score: {similarity_score}")

    else:
        print(f"Movie '{movie_title}' not found in the dataset.")

df, similarity = load_data()
user_input = input("Enter a movie title: ")
recommend_similar_movies(user_input, df, similarity)
