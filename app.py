import streamlit as st
import pandas as pd
import pickle
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Book Recommender", layout="wide")

@st.cache_resource
def load_data():
    books_dict = pickle.load(open('books_dict.pkl', 'rb'))
    df = pd.DataFrame(books_dict)
    vector_matrix = load_npz('vector_matrix.npz')
    return df, vector_matrix

df, vector_matrix = load_data()

st.title("Content-Based Book Recommendation System")

selected_book = st.selectbox(
    "Select or type a book title:",
    df['Book-Title'].values
)

def recommend(book_name):
    index = df[df['Book-Title'] == book_name].index[0]
    similarity_scores = cosine_similarity(vector_matrix[index], vector_matrix).flatten()
    distances = sorted(list(enumerate(similarity_scores)), key=lambda x: x[1], reverse=True)[1:6]
    
    recommended = []
    for i in distances:
        temp_df = df[df.index == i[0]]
        recommended.append({
            'title': temp_df['Book-Title'].values[0],
            'author': temp_df['Book-Author'].values[0],
            'image': temp_df['Image-URL-M'].values[0]
        })
    return recommended

if st.button("Recommend Books"):
    results = recommend(selected_book)
    cols = st.columns(5)
    for col, book in zip(cols, results):
        with col:
            st.image(book['image'], use_container_width=True)
            st.caption(f"**{book['title']}**")
            st.text(f"By {book['author']}")