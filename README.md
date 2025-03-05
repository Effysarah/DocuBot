# DocuBot: Retrieval-Augmented Generation with KB & Web Search

DocuBot is an AI-powered assistant that integrates a knowledge base built from uploaded PDFs, optional web search context, and a Hugging Face inference API to provide enriched answers. This repository contains a Streamlit-based deployment of DocuBot.

## Demo Video

For a quick demonstration of the DocuBot in action, check out the video below:

## Features

- **PDF Upload & Processing:** Upload a PDF to build a knowledge base.
- **In-Memory Knowledge Base:** Extract and store PDF content for context retrieval.
- **Web Search Context (Optional):** Augment responses with live web search results.
- **Hugging Face Inference API:** Generate responses using the DeepSeek-R1-Distill-Qwen-1.5B model.
- **PostgreSQL Integration:** Store conversation history per user (using SQLAlchemy).
- **Streamlit Interface:** User-friendly interface for interacting with the assistant.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/DocuBot.git
   cd DocuBot

Install Dependencies:
Ensure you have Python 3.8 or higher installed. Then, install the required packages:
```bash
pip install -r requirements.txt
```
Set Up PostgreSQL:

Install PostgreSQL and pgAdmin if you haven't already.
Create a new database (e.g., docubot) and update the DB_URL in the code accordingly.
Example connection string:
```bash
postgresql+psycopg2://postgres:your_password@localhost:5432/docubot
```
Running the App
To launch the DocuBot Streamlit app locally, run:
```bash
streamlit run DocuBot.py
```
This command will start the app on your localhost (typically at http://localhost:8501).

## Usage
Enter Credentials:

Provide your Hugging Face API token.
Enter your username (to track conversation history).
Upload PDF:

Optionally, upload a PDF to build your internal knowledge base.
Click "Add PDF to KB" to process and store the PDF content.
Chat with the Assistant:

In the "Chat" tab, enter your question.
Optionally, choose to include web search context by checking the corresponding box.
Click "Get Answer" to receive a response.
View Conversation History:

Switch to the "Conversation History" tab to review your past interactions.
Deployment on Streamlit Cloud
To deploy your DocuBot app on Streamlit Cloud:

Push your repository to GitHub.
Log in to Streamlit Cloud and connect your GitHub repository.
Configure any necessary environment variables (such as your Hugging Face API token).
Deploy the app.
Contributing
Contributions are welcome! Feel free to open issues or submit pull requests with improvements or bug fixes.

## License
This project is licensed under the MIT License.
