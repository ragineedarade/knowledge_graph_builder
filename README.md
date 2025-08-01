# Knowledge Graph Builder with OpenAI Embeddings and Neo4j

This project demonstrates a simple Python script to build a knowledge graph using text embeddings generated by OpenAI's `text-embedding-ada-002` model and store them in a Neo4j graph database. It also includes functionality to perform a basic similarity query based on these embeddings.

## Table of Contents

* [Features](#features)
* [Requirements](#requirements)
* [Setup](#setup)
    * [1. OpenAI API Key](#1-openai-api-key)
    * [2. Neo4j Database](#2-neo4j-database)
    * [3. Project Setup](#3-project-setup)
* [Usage](#usage)
* [How it Works](#how-it-works)
* [Output Example](#output-example)
* [Future Enhancements](#future-enhancements)
* [License](#license)
* [Contact](#contact)

## Features

* **Text Embedding Generation:** Uses OpenAI's `text-embedding-ada-002` model to convert natural language text into high-dimensional numerical vectors (embeddings).
* **Knowledge Graph Creation:** Stores key concepts (nodes) and their relationships in a Neo4j graph database.
* **Embedding Storage:** Embeddings are stored as properties on the Neo4j nodes, enabling semantic operations.
* **Semantic Similarity Search:** Performs cosine similarity queries on stored embeddings to find semantically related concepts within the graph.
* **Environment Variable Management:** Securely handles API keys and database credentials using a `.env` file.

## Requirements

Before you begin, ensure you have the following:

* **Python:** Version 3.6 or higher.
* **OpenAI API Key:** An active OpenAI account with sufficient quota for embedding models. You can get one from [OpenAI Platform](https://platform.openai.com/).
* **Neo4j Database:**
    * A local Neo4j instance (e.g., via [Neo4j Desktop](https://neo4j.com/download/neo4j-desktop/) or [Docker](https://neo4j.com/developer/neo4j-aura-docker/)).
    * **OR** A [Neo4j AuraDB](https://neo4j.com/cloud/aura/) instance (Free Tier is sufficient for testing).

## Setup

### 1. OpenAI API Key

1.  Sign up or log in to the [OpenAI Platform](https://platform.openai.com/).
2.  Navigate to "API keys" under your profile settings.
3.  Create a new secret key and **copy it immediately**, as it will not be shown again.

### 2. Neo4j Database

Choose one of the following options:

* **Neo4j AuraDB (Recommended for beginners):**
    1.  Go to [Neo4j AuraDB](https://neo4j.com/cloud/aura/) and sign up for a free tier account.
    2.  Follow the steps to create a new database instance.
    3.  During creation, you will be given a `Connection URI`, `Username` (usually `neo4j`), and a `Password`. **Crucially, download and save these credentials immediately**, as the password is only shown once.
* **Neo4j Desktop (Local):**
    1.  Download and install [Neo4j Desktop](https://neo4j.com/download/neo4j-desktop/).
    2.  Create a new project and add a local graph database.
    3.  Start the database. The default URI is `bolt://localhost:7687`, username is `neo4j`. You will set a password on first use.
* **Neo4j Docker:**
    1.  Ensure Docker is installed and running.
    2.  Run the following command in your terminal (replace `your_neo4j_password`):
        ```bash
        docker run \
            --publish=7474:7474 --publish=7687:7687 \
            --env=NEO4J_AUTH=neo4j/your_neo4j_password \
            --name=my_neo4j_instance \
            neo4j:latest
        ```
        The URI will be `bolt://localhost:7687`, username `neo4j`, and the password you specified.

### 3. Project Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your_github_username/your_repo_name.git](https://github.com/your_github_username/your_repo_name.git)
    cd your_repo_name
    ```
    (Replace `your_github_username` and `your_repo_name` with your actual GitHub details.)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install openai neo4j python-dotenv
    ```

4.  **Create a `.env` file:**
    In the root directory of your project (the same directory as `app.py`), create a new file named **`.env`** (make sure it has no file extension like `.txt`). Add your credentials to this file:

    ```env
    OPENAI_API_KEY="sk-YOUR_OPENAI_API_KEY_HERE"
    NEO4J_URI="bolt://localhost:7687" # Use your Aura URI if applicable (e.g., "neo4j+s://xxxx.databases.neo4j.io:7687")
    NEO4J_USERNAME="neo4j"
    NEO4J_PASSWORD="YOUR_NEO4J_PASSWORD_HERE"
    ```
    * **IMPORTANT:** Replace the placeholder values with your actual OpenAI API key and Neo4j credentials.
    * **SECURITY NOTE:** Never commit your `.env` file to version control. It's automatically added to `.gitignore` by default in many templates, but always double-check.

## Usage

Once all setup steps are complete, run the Python script:

```bash
python app.py
