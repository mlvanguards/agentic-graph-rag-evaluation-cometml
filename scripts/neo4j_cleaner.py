from neo4j import GraphDatabase
from src.config.settings import Settings

settings = Settings()

class Neo4jCleaner:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def delete_all_data(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("All nodes and relationships have been deleted.")

    def delete_constraints_and_indexes(self):
        with self.driver.session() as session:
            # Drop all constraints
            for constraint in session.run("SHOW CONSTRAINTS"):
                session.run(f"DROP CONSTRAINT {constraint['name']}")

            # Drop all indexes
            for index in session.run("SHOW INDEXES"):
                session.run(f"DROP INDEX {index['name']}")

            print("All constraints and indexes have been dropped.")


def main():
    cleaner = Neo4jCleaner(uri=settings.neo4j_uri, user=settings.neo4j_user, password=settings.neo4j_password)

    try:
        cleaner.delete_all_data()
        cleaner.delete_constraints_and_indexes()
    finally:
        cleaner.close()


if __name__ == "__main__":
    main()