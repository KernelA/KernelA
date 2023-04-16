import json
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Category:
    title: str

    @staticmethod
    def from_dict(data: dict):
        return Category(**data)


@dataclass
class CategoryCollection:
    categories: Dict[str, Category]

    @staticmethod
    def from_json(path_to_file: str):
        with open(path_to_file, "rb") as f:
            cat_info = json.load(f)

        categories = {cat_id: Category.from_dict(
            info) for cat_id, info in cat_info.items()}

        return CategoryCollection(categories)

    def get_title_by_id(self, uniq_id: str):
        return self.categories[uniq_id].title
