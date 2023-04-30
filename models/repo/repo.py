import re
from dataclasses import dataclass
from typing import List

from github.Repository import Repository as ExternalRepo

CAT_REGEX = re.compile(r"^cat-\d+$")
EXCLUDE_TOPIC = "gh-prof-exclude"


@dataclass
class Repository:
    name: str
    desc: str
    link: str
    category: str
    languages: list[str]
    is_filter: bool = False

    @staticmethod
    def from_model(repo: ExternalRepo):
        category = "unknown"
        is_filter = False

        for topic in repo.topics:
            if topic == EXCLUDE_TOPIC:
                is_filter = True
            elif re.match(CAT_REGEX, topic) is not None:
                category = topic

        return Repository(name=repo.name,
                          desc="" if repo.description is None else repo.description.strip(),
                          link=repo.html_url,
                          category=category,
                          is_filter=is_filter,
                          languages=sorted(repo.get_languages().keys()))


@dataclass
class RepoCollection:
    all_repos: List[Repository]

    def __post_init__(self):
        self.all_repos.sort(key=lambda repo: repo.category)
