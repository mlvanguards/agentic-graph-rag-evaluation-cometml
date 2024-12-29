from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

@dataclass
class Paper:
    id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    submit_date: datetime
    update_date: Optional[datetime] = None

    @classmethod
    def from_db_record(cls, record: dict) -> 'Paper':
        return cls(
            id=record.get('id'),
            title=record.get('title'),
            abstract=record.get('abstract'),
            authors=record.get('authors', []),
            categories=record.get('categories', []),
            submit_date=record.get('submit_date'),
            update_date=record.get('update_date')
        )

    def to_string(self) -> str:
        return f"""Title: {self.title}
Abstract: {self.abstract}
Authors: {', '.join(self.authors) if self.authors else 'No authors listed'}
Categories: {', '.join(self.categories) if self.categories else 'No categories listed'}
Submitted on: {self.submit_date}
Updated on: {self.update_date if self.update_date else 'N/A'}"""
