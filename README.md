# Content-Based Book Recommendation System

### What is this app?

This is a machine learning web application implementing a strict content-based filtering architecture. 
Unlike collaborative filtering systems that blindly rely on user rating history and suffer from the item cold-start problem, this system evaluates the inherent attributes of the items themselves. It processes book metadata to autonomously recommend highly similar items even if the books have zero user ratings.

Live Demo: [https://content-based-book-recommender-gdvqruekmjdcmmgaln53nu.streamlit.app/]

### Architecture & Tech Stack

* Frontend: Streamlit
* Backend: Python
* Data Processing: Pandas, NumPy
* Vectorization: TF-IDF (Scikit-Learn)
* Similarity Engine: Cosine Similarity
* Memory Optimization: Scipy Sparse Matrices

### The Workflow

When a user selects a book title, the application executes the following pipeline:

* **Feature Engineering:** The system concatenates the target book's metadata (Author, Publisher, Year of Publication) into a standardized, lowercased string of tags.
* **Vectorization:** The system converts the text tags into a mathematical vector space using Term Frequency-Inverse Document Frequency (TF-IDF). This scales down high-frequency generic words while assigning higher weights to unique identifiers like specific authors.
* **Sparse Matrix Optimization:** To prevent cloud memory limits from being exceeded by a dense 1.8 GB matrix, the deployment environment loads a compressed sparse feature matrix (.npz) weighing less than 10 MB. 
* **Dynamic Similarity Calculation:** The system computes the 1-to-N cosine similarity between the selected book's vector and the 15,000 other vectors in the sparse matrix on the fly during inference.
* **Result Extraction:** The system sorts the resulting array, slices the top 5 highest mathematical matches (excluding the target index 0), and maps them back to the original DataFrame to retrieve titles, authors, and cover image URLs for the UI.

### Key Features

* **Zero Cold-Start Dependency:** Instantly recommends items without requiring historical user interaction or rating datasets. 
* **Memory-Optimized Inference:** Reduces production RAM consumption by over 99% by discarding pre-calculated dense matrices in favor of compressed sparse matrices and runtime vector operations.
* **Visual Data Delivery:** Returns structured metadata including image paths, allowing the frontend to render clean, interactive UI cards rather than raw text outputs.

### Setup instructions

1. Clone the Repository & install Dependencies:
    ```bash
    git clone [https://github.com/your-username/content-based-book-recommender.git](https://github.com/your-username/content-based-book-recommender.git)
    cd content-based-book-recommender
    pip install -r requirements.txt
    ```

2. Run the Application:
    ```bash
    streamlit run app.py
    ```
    Open http://localhost:8501 in your browser.
