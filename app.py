from flask import Flask, render_template, request, jsonify
import ollama
import chromadb
from datetime import datetime

app = Flask(__name__)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("qa_collection")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    user_question = data['question'].strip()
    force_new = data.get('force_new', False)
    max_words = 80

    skip_memory = force_new or any(word in user_question.lower() for word in ["random", "new", "generate", "fresh"])

    # Query stored answers
    if not skip_memory:
        try:
            results = collection.query(query_texts=[user_question], n_results=1)
            if results['documents'] and results['documents'][0]:
                stored_question = results['metadatas'][0][0]['question']
                stored_answer = results['documents'][0][0]
                stored_time = results['metadatas'][0][0]['timestamp']
                similarity = results.get('distances', [[1]])[0][0]
                match_percent = round((1 - similarity) * 100, 2)

                if match_percent > 75 and len(user_question) > 5:
                    return jsonify({
                        "answer": stored_answer,
                        "match_percent": match_percent,
                        "matched_question": stored_question,
                        "timestamp": stored_time,
                        "from_memory": True
                    })
        except Exception:
            pass

    # Generate a new response with Ollama
    try:
        response = ollama.chat(model="llama3", messages=[{"role": "user", "content": user_question}])
        answer = response['message']['content'].strip()
    except Exception as e:
        answer = f"Error: Could not generate a response. ({e})"

    # Apply word limit
    words = answer.split()
    if len(words) > max_words:
        answer = " ".join(words[:max_words]) + "..."

    # Store in Chroma
    try:
        collection.add(
            documents=[answer],
            metadatas=[{
                "question": user_question,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }],
            ids=[str(datetime.now().timestamp())]
        )
    except Exception:
        pass

    return jsonify({
        "answer": answer,
        "match_percent": 0,
        "matched_question": None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "from_memory": False
    })

if __name__ == '__main__':
    app.run(debug=True)
