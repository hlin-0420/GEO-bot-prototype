from rapidfuzz import process
from sentence_transformers import SentenceTransformer, util

# Function to check if a question has a similar wording in expected_answers_dict
def is_similar_question(question, expected_answers, threshold=0.80):
    print(f"question: {question}")
    print(f"expected answer: {expected_answers}")
    match, score, _ = process.extractOne(question, expected_answers, score_cutoff=threshold)
    if match is None:
        return False
    print(f"score: {score}")
    return score >= threshold  # Returns True if similarity score is above threshold

# Function to find the most semantically similar expected answer
def find_best_match(question, expected_answers_dict):
    if not question or not expected_answers_dict:
        return ""  # Return empty string if no valid input

    expected_questions = list(expected_answers_dict.keys())
    
    # Load a pre-trained sentence transformer model
    model = SentenceTransformer("all-MiniLM-L6-v2")  # Lightweight and efficient

    # Compute embeddings
    question_embedding = model.encode(question, convert_to_tensor=True)
    expected_embeddings = model.encode(expected_questions, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.pytorch_cos_sim(question_embedding, expected_embeddings)[0]

    # Find the best match
    best_match_idx = similarities.argmax().item()
    best_match_score = similarities[best_match_idx].item()

    # Set a similarity threshold (adjust if needed)
    threshold = 0.75  
    if best_match_score < threshold:
        return ""  # No match found above the threshold

    return expected_answers_dict[expected_questions[best_match_idx]]

def calculate_semantic_similarity(text1, text2, model_name='all-MiniLM-L6-v2'):
    """
    Computes the semantic similarity between two text extracts using Sentence-BERT.

    Parameters:
    text1 (str): The first text (expected response).
    text2 (str): The second text (chatbot response).
    model_name (str): Name of the pre-trained SentenceTransformer model.

    Returns:
    float: Cosine similarity score (range: 0 to 1, where 1 means identical meaning).
    """
    model = SentenceTransformer(model_name)

    # Generate embeddings
    embeddings = model.encode([text1, text2], convert_to_tensor=True)

    # Compute cosine similarity
    similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    
    return similarity_score

# Function to replace invalid values with an empty string
def clean_dataframe(df):
    """
    Replace NaN, None, or other invalid values with an empty string in the given DataFrame.
    """
    return df.fillna("").replace({None: ""})  # Replace NaN and None with empty string