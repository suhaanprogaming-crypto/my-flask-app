from flask import Flask, render_template, request, jsonify, session
import ollama
import chromadb
from datetime import datetime

app = Flask(__name__)
app.secret_key = "supersecret"

MODEL = "llama3"
MAX_WORDS = 250

# Chroma DB for memory
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("qa_collection")

@app.route('/')
def index():
    if 'conversation' not in session:
        session['conversation'] = [{"role": "system", "content": "You are a helpful AI assistant."}]
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    user_question = data.get('question', '').strip()
    force_new = data.get('force_new', False)

    if not user_question:
        return jsonify({"answer": "Please enter a question.", "from_memory": False})

    skip_memory = force_new or any(word in user_question.lower() for word in ["random", "new", "generate", "fresh"])

    # Check Chroma memory
    if not skip_memory:
        try:
            results = collection.query(query_texts=[user_question], n_results=1)
            if results['documents'] and results['documents'][0]:
                stored_question = results['metadatas'][0][0]['question']
                stored_answer = results['documents'][0][0]
                stored_time = results['metadatas'][0][0]['timestamp']
                similarity = results.get('distances', [[1]])[0][0]
                match_percent = round((1 - similarity) * 100, 2)

                if match_percent > 70 and len(user_question) > 5:
                    return jsonify({
                        "answer": stored_answer,
                        "match_percent": match_percent,
                        "matched_question": stored_question,
                        "timestamp": stored_time,
                        "from_memory": True
                    })
        except Exception:
            pass

    # Append user message to conversation
    conversation = session.get('conversation', [])
    conversation.append({"role": "user", "content": user_question})

    # Generate response from Ollama llama3
    try:
        response = ollama.chat(model=MODEL, messages=conversation)
        answer = response.get('message', {}).get('content', '').strip() or "No response generated."
    except Exception as e:
        answer = f"Error: Could not generate a response. ({e})"

    conversation.append({"role": "assistant", "content": answer})
    session['conversation'] = conversation

    # Store in Chroma
    try:
        collection.add(
            documents=[answer],
            metadatas=[{"question": user_question, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}],
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
