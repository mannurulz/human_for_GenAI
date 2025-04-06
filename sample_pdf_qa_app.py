import openai
import os
from dotenv import load_dotenv
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() or "" #handle if a page has no text.
    except Exception as e:
        print(f"Error processing PDF: {e}")
    return text

def create_embeddings(texts):
    """Creates sentence embeddings for the given texts."""
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(texts)
    return embeddings

def create_index(embeddings):
    """Creates a FAISS index for efficient similarity search."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def find_relevant_context(query, embeddings, index, texts, k=3):
    """Finds the most relevant context for the given query."""
    query_embedding = create_embeddings([query])
    distances, indices = index.search(np.array(query_embedding), k)
    relevant_context = [texts[i] for i in indices[0]]
    return "\n".join(relevant_context)

def answer_question(question, context):
    """Answers a question using the OpenAI API and the provided context."""
    try:
        prompt = f"Answer the following question based on the provided context:\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        response = openai.Completion.create(
            engine="text-davinci-003", #or gpt-3.5-turbo-instruct or a newer model
            prompt=prompt,
            max_tokens=500,
            n=1,
            stop=None,
            temperature=0.3, #lower temperature for more factual answers
        )
        answer = response.choices[0].text.strip()
        return answer
    except Exception as e:
        return f"Error: {e}"

def main():
    """Main function to run the PDF question-answering app."""
    pdf_path = input("Enter the path to your PDF file: ")
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        print("Could not extract text from PDF.")
        return

    # Split the PDF text into smaller chunks (e.g., sentences or paragraphs)
    texts = pdf_text.split("\n")  # Simple split by newline, you can refine this.
    texts = [text for text in texts if text.strip()] #remove empty strings.

    embeddings = create_embeddings(texts)
    index = create_index(embeddings)

    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break

        relevant_context = find_relevant_context(question, embeddings, index, texts)
        answer = answer_question(question, relevant_context)
        print("\nAnswer:\n", answer, "\n")

if __name__ == "__main__":
    main()
