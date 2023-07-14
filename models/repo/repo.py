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
    categories: list[str]
    languages: list[str]
    is_filter: bool = False

    @staticmethod
    def from_model(repo: ExternalRepo):
        categories = []
        is_filter = False

        for topic in repo.topics:
            if topic == EXCLUDE_TOPIC:
                is_filter = True
            elif re.match(CAT_REGEX, topic) is not None:
                categories.append(topic)

        if not categories:
            categories = ["unknown"]

        return Repository(name=repo.name,
                          desc="" if repo.description is None else repo.description.strip(),
                          link=repo.html_url,
                          categories=categories,
                          is_filter=is_filter,
                          languages=sorted(repo.get_languages().keys()))


@dataclass
class RepoCollection:
    all_repos: List[Repository]

    def __post_init__(self):
        self.all_repos.sort(key=lambda repo: repo.categories)
