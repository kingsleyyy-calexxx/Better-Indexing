import os
import re
import numpy
import pymongo
import nltk
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["Search_Indexing_New"]
collection = db["Words"]
indexed_pages_collection = db["Indexed pages"]

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()
global_doc_number = 0

# Function to extract text from HTML file
def extract_text_from_html_with_priority(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    tag_priority = {"title": 1, "h1": 2, "h2": 3, "h3": 4, "h4": 5, "h5": 6, "h6": 7, "p": 8, "div": 9, "span": 10}

    text = ""
    for tag_name in tag_priority:
        tags = soup.find_all(tag_name)
        for tag in tags:
            text += tag.get_text(separator=" ", strip=True) + " "

    return text.lower()

# Function to perform stemming and remove stop words
def preprocess_text(text):
    words = nltk.word_tokenize(text)
    words = [ps.stem(word) for word in words if word.isalnum() and word not in stop_words]
    return words

# Function to calculate TF-IDF
def calculate_tfidf(documents):
    # Check if any documents are present
    if not any(documents):
        return None, None

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names

# Function to index and insert into MongoDB
def index_and_insert_with_priority(directory, link, tag_priority):
    global global_doc_number  # Declare global variable

    tfidf_calculated = False  # Flag to track TF-IDF calculation

    for root, dirs, files in os.walk(directory):
        for file_name in sorted(files):  # Sorting files to ensure consistent order
            if file_name.endswith(".html"):
                global_doc_number += 1  # Increment document number for each processed file
                doc_number = global_doc_number  # Assign the current document number
                html_file_path = os.path.join(root, file_name)
                with open(html_file_path, "r", encoding="utf-8") as file:
                    html_content = file.read()

                # Extract text from HTML with tag priorities
                text = extract_text_from_html_with_priority(html_content)

                # Preprocess text (stemming, lowercasing)
                words = preprocess_text(text)

                # Calculate word frequency
                word_freq = Counter(words)

                # Calculate TF-IDF
                tfidf_matrix, feature_names = calculate_tfidf([text])

                if tfidf_matrix is not None and feature_names is not None:
                    tfidf_calculated = True

                    # Insert into MongoDB only if the page is not indexed
                    if not indexed_pages_collection.find_one({"link": link, "file_name": file_name}):
                        for word, freq in word_freq.items():
                            # Find the index of the word in feature_names
                            word_index = numpy.where(feature_names == word)[0]

                            if word_index.size > 0:  # Check if the word is found in feature_names
                                tfidf_value = tfidf_matrix[0, word_index[0]]

                                info = {
                                    "doc_no": doc_number,
                                    "frequency": freq,
                                    "priority": tag_priority.get(word, 11),
                                    # Set priority based on your criteria (default to 11 if not found)
                                    "tf-idf": tfidf_value,
                                    "link": link
                                }

                                # Update MongoDB document
                                collection.update_one(
                                    {"word": word},
                                    {
                                        "$push": {"info_list": info},
                                        "$setOnInsert": {"word": word}
                                    },
                                    upsert=True
                                )

                        # Mark the page as indexed
                        indexed_pages_collection.insert_one({"link": link, "file_name": file_name, "doc_no": doc_number})

    # Display TF-IDF calculation message
    if not tfidf_calculated:
        print(f"TF-IDF not calculated for website: {link}")

# Main indexing loop
folder_path = r'C:\Users\HP\crawler-bucket\Folder_1'
for folder_name in os.listdir(folder_path):
    folder_link = os.path.splitext(folder_name)[0]  # Extract website link from folder name
    folder_directory = os.path.join(folder_path, folder_name)

    # Index and insert into MongoDB with tag priorities
    tag_priority = {"title": 1, "h1": 2, "h2": 3, "h3": 4, "h4": 5, "h5": 6, "h6": 7, "p": 8, "div": 9, "span": 10}
    index_and_insert_with_priority(folder_directory, folder_link, tag_priority)

# Display indexing completion message
print("Indexing complete.")
