
# Web-Based Chatbot for Translating Natural Language to Cypher Queries(Assessment Task)

a web-based chatbot that translates natural language questions into Cypher queries to interact with a Neo4j database

## Features

1. **Natural Language to Cypher Translation**  
- Converts user queries into Cypher queries using Ollama meta-llama/Llama-3.2-1B.  

2. **Neo4j Integration**  
- Executes Cypher queries on a locally hosted Neo4j database and retrieves relevant results.  

3. **Web-Based Interface**  
- Provides a user-friendly interface for users to input queries and view results seamlessly.  


## Project Structure

```

__pycache__
env
templates/
	index.html
.gitignore
app.py
llmintegrator.py
requirements.txt
Readme.md

```
## Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/neelamnagarajgithub/GSOC_NRNB_25_DBretina_Chatbot
    cd <repository-directory>
    ```

2. **Create a virtual environment:**

    ```sh
    python3 -m venv env
    source env/bin/activate
    ```

3. **Install dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

## Running the Application

1. **Initializing the LLM locally**

	Navgiate to [Ollama](https://ollama.com/download) choose your os and install on your system. Then run the folling command to pull llama 3.2:1b

	```
	ollama run llama3.2:1b
	```

2. **Set up Neo4j:**

    Ensure you have a running Neo4j instance and set the environment variables in [llmintegrator.py](http://_vscodecontentref_/3):

    ```python
    os.environ["NEO4J_URI"] = "bolt://localhost:7687"
    os.environ["NEO4J_USERNAME"] = "your-username"
    os.environ["NEO4J_PASSWORD"] = "your-password"
    ```

3. **Run the FastAPI server:**

    ```sh
    uvicorn app:app --reload
    ```

    The server will start at `http://127.0.0.1:8000`.

## API Endpoints

- **POST /query**

    Query the graph database.

    **Request:**

    ```json
    {
        "query": "What was the cast of the Toy Story?"
    }
    ```

    **Response:**

    ```json
    {
        "response": "<response from the language model>"
    }
    ```

## Files

- **`llmintegrator.py`**: Contains the main logic for connecting to Neo4j, importing movie data, and setting up the GraphCypherQAChain.
- **`app.py`**: Entry point for the FastAPI application.
- **`templates/index.html`**: HTML template for the application.

## License

This project is licensed under the MIT License.

