import json
import argparse
from typing import List, Dict, Any


def preprocess_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Preprocess the arXiv papers data.

    Args:
        data: List of dictionaries containing paper information

    Returns:
        List of processed paper dictionaries
    """
    processed_data = []
    for paper in data:
        processed_paper = {
            'id': paper['id'],
            'title': paper['title'],
            'abstract': paper['abstract'],
            'categories': paper['categories'].split(),
            'authors': [' '.join(author).strip()
                        for author in paper['authors_parsed']],
            'submit_date': paper['versions'][0]['created'],
            'update_date': paper['update_date']
        }
        processed_data.append(processed_paper)
    return processed_data


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Preprocess arXiv papers data')
    parser.add_argument('--input', type=str, required=True,
                        help='Input JSON file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output JSON file path')

    # Parse arguments
    args = parser.parse_args()

    # Read input file
    print(f"Reading data from {args.input}...")
    try:
        with open(args.input, 'r') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file {args.input} not found")
        return
    except json.JSONDecodeError:
        print(f"Error: Input file {args.input} is not valid JSON")
        return

    # Process the data
    print("Processing data...")
    processed_data = preprocess_data(raw_data)

    # Save processed data
    print(f"Saving processed data to {args.output}...")
    try:
        with open(args.output, 'w') as f:
            json.dump(processed_data, f, indent=2)
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error saving output file: {str(e)}")
        return


if __name__ == "__main__":
    main()
