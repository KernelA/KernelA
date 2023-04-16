from dataclasses import dataclass, field
from typing import Dict, List

import pytablereader as ptr

from ..catgeory import CategoryCollection
from ..repo import Repository

RowType = List[str]


@dataclass
class Table:
    title: str

    headers: List[str]
    rows: List[RowType]

    def __post_init__(self):
        if self.headers and self.rows:
            if len(self.headers) != len(self.rows[0]):
                raise RuntimeError(
                    "Number of header does not equal number of columns")

        if self.rows:
            if not all(map(lambda x: x == len(self.rows[0]), map(len, self.rows))):
                raise RuntimeError("Not all rows have same number of columns")

        self.rows.sort(key=lambda x: x[0])

    @staticmethod
    def from_markdown(path_to_file: str):
        title = None

        table_lines = []

        with open(path_to_file, "r", encoding="utf-8") as f:
            for line in map(str.strip, f):
                if line.startswith("#"):
                    title = line[1:].strip()

                if line and title:
                    table_lines.append(line)

        if not title:
            raise ValueError(
                f"cannot find header H1 in the '{path_to_file}'. Please specify H1 header")

        headers = []
        rows = []

        for table_data in ptr.MarkdownTableTextLoader("\n".join(table_lines)).load():
            if headers:
                raise RuntimeError(
                    f"Expected only one table per file but found at least 2: '{path_to_file}'")

            headers = list(table_data.headers)
            rows = list(map(list, table_data.rows))

        link_index = headers.index("Link")
        name_index = headers.index("Name")

        for i in range(len(rows)):
            rows[i][name_index] = f"[{rows[i][name_index]}]({rows[i][link_index]})"
            del rows[i][link_index]

        del headers[link_index]

        return Table(title=title, headers=headers, rows=rows)

    def merge_inplace(self, other_table: "Table"):
        if self.headers != other_table.headers:
            raise ValueError("Cannot merge tables with different headers")
        self.rows.extend(other_table.rows)

    @staticmethod
    def from_repo(repo: Repository, cat_collections: CategoryCollection):
        title = cat_collections.get_title_by_id(repo.category)

        return Table(
            title,
            ["Name", "Description", "Languages"],
            [
                [
                    f"[{repo.name}]({repo.link})",
                    repo.desc,
                    ", ".join(repo.languages)]
            ]
        )


@dataclass
class TableCollection:
    tables: Dict[str, Table] = field(default_factory=dict)

    def add_table(self, new_table: Table):
        table = self.tables.get(new_table.title, None)

        if table is None:
            self.tables[new_table.title] = new_table
        else:
            table.merge_inplace(new_table)
