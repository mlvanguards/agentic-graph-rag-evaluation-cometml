1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   
2. **Download Dataset**
   - Visit [arXiv Dataset on Kaggle](https://www.kaggle.com/datasets/Cornell-University/arxiv)
   - Download and extract the dataset to your project directory

2. **Preprocess the Data**
   - Create a Python script (e.g., `run_preprocessing.py`) with:
     ```python
     import json
     from scripts.preprocess import preprocess_data
     
     # Load the JSON file
     with open('path_to_downloaded_arxiv.json', 'r') as f:
         raw_data = json.load(f)
     
     # Preprocess the data
     processed_data = preprocess_data(raw_data)
     
     # Save the processed data
     with open('processed_arxiv.json', 'w') as f:
         json.dump(processed_data, f)
     ```
   - Run the script:
    
     python run_preprocessing.py
    
3. **Configure Credentials**
   - Create a .env file in the project root
   - Add the following credetials:
     -    NEO4J_URI=your_neo4j_uri
     NEO4J_USER=your_username
     NEO4J_PASSWORD=your_password
     COMET_API_KEY=your_comet_key
     OPENAI_API_KEY=your_openai_key

4. **Ingest data in Neo4j**
   - Run the script:
    
    python research_agent/components/database/ingest.py
5. **Start the applications**
   - streamlit run main.py
