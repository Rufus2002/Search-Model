import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import streamlit as st

# Load a model that supports embedding generation from Hugging Face
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Sample course data
courses = [
  {"title": "Introduction to Python", "description": "Learn Python basics", "level": "beginner"},
  {"title": "Data Structures and Algorithms", "description": "Master the fundamental building blocks of computer science", "level": "intermediate"},
  {"title": "Web Development with HTML, CSS, and JavaScript", "description": "Learn the basics of web development and design", "level": "beginner"},
  {"title": "Machine Learning for Beginners", "description": "An introduction to the basics of machine learning", "level": "beginner"},
  {"title": "Advanced Machine Learning", "description": "Learn complex machine learning algorithms and techniques", "level": "advanced"},
  {"title": "Introduction to Cyber Security", "description": "Get familiar with the essentials of cybersecurity", "level": "beginner"},
  {"title": "Network Security", "description": "Learn how to protect networks from cyber threats", "level": "intermediate"},
  {"title": "Ethical Hacking Basics", "description": "Learn ethical hacking and penetration testing techniques", "level": "beginner"},
  {"title": "AI for Data Science", "description": "Learn how to implement AI algorithms for data analysis", "level": "intermediate"},
  {"title": "Deep Learning with TensorFlow", "description": "Understand and implement deep learning models using TensorFlow", "level": "advanced"},
  {"title": "Introduction to IoT", "description": "Learn the fundamentals of the Internet of Things (IoT)", "level": "beginner"},
  {"title": "IoT with Arduino", "description": "Create IoT applications using Arduino boards", "level": "intermediate"},
  {"title": "Cyber Security Fundamentals", "description": "Get an overview of cybersecurity principles and practices", "level": "beginner"},
  {"title": "Cloud Computing and Security", "description": "Understand cloud computing principles and how to secure cloud infrastructure", "level": "intermediate"},
  {"title": "Generative AI for Beginners", "description": "Learn about generative AI models and their applications", "level": "beginner"},
  {"title": "Ethical Hacking: Offensive Techniques", "description": "Learn advanced offensive security techniques for ethical hacking", "level": "advanced"},
  {"title": "Introduction to C Programming", "description": "Learn C programming language fundamentals", "level": "beginner"},
  {"title": "IoT Security", "description": "Learn how to secure IoT devices and networks", "level": "intermediate"},
  {"title": "Big Data Analytics", "description": "Learn how to handle and analyze large data sets", "level": "intermediate"},
  {"title": "Mobile App Development with Android", "description": "Learn to build mobile applications for Android", "level": "beginner"},
  {"title": "Advanced Ethical Hacking", "description": "Learn advanced penetration testing and ethical hacking skills", "level": "advanced"},
  {"title": "Introduction to Information Technology", "description": "Understand the basics of information technology and systems", "level": "beginner"},
  {"title": "Introduction to Robotics", "description": "Learn the basics of robotics and automation systems", "level": "beginner"},
  {"title": "Natural Language Processing with Python", "description": "Learn how to process and analyze text data using Python", "level": "intermediate"},
  {"title": "Data Science and AI", "description": "Combine data science techniques with AI algorithms", "level": "intermediate"},
  {"title": "Computer Vision Basics", "description": "Learn the fundamentals of computer vision and image processing", "level": "beginner"},
  {"title": "Machine Learning for IoT", "description": "Implement machine learning models for IoT applications", "level": "intermediate"},
  {"title": "Artificial Intelligence Fundamentals", "description": "An introduction to the concepts and applications of AI", "level": "beginner"},
  {"title": "Cloud Security Essentials", "description": "Learn how to secure cloud-based environments", "level": "intermediate"},
  {"title": "Arduino for Beginners", "description": "Learn to build projects using Arduino", "level": "beginner"},
  {"title": "Blockchain Technology", "description": "Learn about blockchain and its applications", "level": "intermediate"},
  {"title": "Advanced Deep Learning", "description": "Explore advanced deep learning models and techniques", "level": "advanced"},
  {"title": "AI in Healthcare", "description": "Explore the applications of AI in the healthcare industry", "level": "intermediate"},
  {"title": "Ethical Hacking: Web Application Security", "description": "Learn to test and secure web applications", "level": "intermediate"},
  {"title": "Introduction to Cloud Computing", "description": "Learn about cloud computing services and models", "level": "beginner"},
  {"title": "Python for Data Science", "description": "Learn Python programming with a focus on data science", "level": "beginner"},
  {"title": "Cyber Forensics", "description": "Learn digital forensics techniques for investigating cybercrimes", "level": "intermediate"},
  {"title": "Embedded Systems Design", "description": "Learn how to design embedded systems with microcontrollers", "level": "intermediate"},
  {"title": "IoT with Raspberry Pi", "description": "Learn to build IoT systems using Raspberry Pi", "level": "intermediate"},
  {"title": "Introduction to Ethical Hacking", "description": "Begin your ethical hacking journey by learning security basics", "level": "beginner"},
  {"title": "Data Mining Techniques", "description": "Learn how to extract useful information from large datasets", "level": "intermediate"},
  {"title": "AI Chatbot Development", "description": "Learn to build conversational AI systems", "level": "intermediate"},
  {"title": "AI for Cyber Security", "description": "Explore AI applications in enhancing cybersecurity", "level": "intermediate"},
  {"title": "Deep Learning for Computer Vision", "description": "Implement deep learning models for computer vision tasks", "level": "advanced"},
  {"title": "Generative Adversarial Networks (GANs)", "description": "Learn how GANs work and create new images and data", "level": "advanced"},
  {"title": "Quantum Computing Basics", "description": "An introduction to quantum computing and its potential", "level": "beginner"},
  {"title": "IoT Data Management", "description": "Learn how to handle and process data in IoT systems", "level": "intermediate"},
  {"title": "Introduction to Artificial Neural Networks", "description": "Understand the basics of artificial neural networks", "level": "beginner"},
  {"title": "Introduction to Digital Marketing and Analytics", "description": "Learn how to use digital marketing tools and analytics", "level": "beginner"},
  {"title": "AI for Robotics", "description": "Apply AI algorithms to robotics and automation", "level": "intermediate"},
  {"title": "Raspberry Pi for Beginners", "description": "Learn to use Raspberry Pi for a variety of projects", "level": "beginner"},
  {"title": "Autonomous Vehicles and AI", "description": "Learn how AI is applied to self-driving car technologies", "level": "advanced"},
  {"title": "Introduction to Cloud-Based Machine Learning", "description": "Learn to build and deploy ML models in the cloud", "level": "intermediate"}
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
k = 5  # Number of nearest neighbors
distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)

# Display the results
st.header("Search Results:")
for idx in indices[0]:
    print(st.write(f"{courses[idx]['title']}"))
    print(st.write(f"{"Description": {courses[idx]['description']}"))
    
