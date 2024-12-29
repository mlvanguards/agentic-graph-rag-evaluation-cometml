import json
import multiprocessing
import time
from dotenv import load_dotenv
from src.config.settings import Settings
from src.components.database.neo4j_ingestion import OptimizedNeo4jIngestor, worker

load_dotenv()

def ingest_data_parallel(uri: str, user: str, password: str, data, batch_size: int = 1000, num_processes: int = 4):
    total = len(data)
    batches = [data[i:i + batch_size] for i in range(0, total, batch_size)]

    with multiprocessing.Pool(num_processes) as pool:
        results = []
        for batch in batches:
            result = pool.apply_async(worker, (uri, user, password, batch))
            results.append(result)

        for i, result in enumerate(results):
            result.get()  # Wait for the batch to complete
            print(f"Ingested {min((i + 1) * batch_size, total)}/{total} papers")


def main():
    settings = Settings()
    uri = settings.neo4j_uri
    user = settings.neo4j_user
    password = settings.neo4j_password

    # Initialize and create constraints
    ingestor = OptimizedNeo4jIngestor(uri, user, password)
    ingestor.create_constraints()
    ingestor.close()

    # Load data
    with open('processed_data.json', 'r') as f:
        processed_data = json.load(f)

    # Ingest in parallel
    start_time = time.time()
    ingest_data_parallel(uri, user, password, processed_data, batch_size=1000, num_processes=4)
    end_time = time.time()

    print(f"Total ingestion time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
