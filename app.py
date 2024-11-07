import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Load a model that supports embedding generation from Hugging Face
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Sample course data
courses = [
    {"title": "Introduction to Python", "description": "Learn Python basics", "level": "beginner"},
    {"title": "Advanced Python for Data Science", "description": "Python for advanced data science topics", "level": "advanced"},
    {"title": "Data Science with Python", "description": "Learn data science concepts with Python", "level": "intermediate"},
    {"title": "Machine Learning Basics", "description": "Introduction to machine learning", "level": "beginner"}
]

# Function to generate embeddings using Hugging Face model
def get_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Average pooling
    return embeddings.squeeze().numpy()  # Convert to NumPy array

# Generate embeddings for each course description
course_embeddings = []
for course in courses:
    embedding = get_embeddings(course["description"])
    course_embeddings.append(embedding)

# Convert embeddings list to a NumPy array for FAISS
embeddings_array = np.array(course_embeddings).astype('float32')

# Set up the FAISS vector index
embedding_dim = embeddings_array.shape[1]
index = faiss.IndexFlatL2(embedding_dim)
index.add(embeddings_array)

print("FAISS index created and embeddings added successfully.")

# Streamlit app interface
st.title("Course Search and Filter")

# Search box for query input
query_text = st.text_input("Enter keywords to search for courses:")

# Filter options
selected_level = st.selectbox("Select Level", options=["All", "Beginner", "Intermediate", "Advanced"])
selected_domain = st.selectbox("Select Domain", options=["All", "Programming", "Data Science", "Machine Learning"])


# Example search query
query_embedding = get_embeddings(query_text)

# Perform the FAISS search
k = 2  # Number of nearest neighbors
distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)

# Display the results
print("Search Results:")
for idx in indices[0]:
    print(f"Course Title: {courses[idx]['title']}, Description: {courses[idx]['description']}")
