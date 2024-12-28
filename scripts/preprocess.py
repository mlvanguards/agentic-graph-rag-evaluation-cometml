from typing import List, Dict, Any

def preprocess_data(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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