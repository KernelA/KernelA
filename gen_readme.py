import json
import os
import pickle

import configargparse
from github import Github, NamedUser
from jinja2 import Environment, FileSystemLoader, select_autoescape
from tqdm.auto import tqdm

from models import (CategoryCollection, RepoCollection, Repository, Table,
                    TableCollection)


def load_categories(path_to_json: str):
    return CategoryCollection.from_json(path_to_json)


def load_exclusion_repo_list(path_to_file: str) -> set:
    with open(path_to_file, "rb") as f:
        exclusion_list = set(json.load(f).keys())

    return exclusion_list


def load_static_tables(table_collection: TableCollection, path_to_dir: str):
    for path in os.scandir(path_to_dir):
        if os.path.splitext(path)[1] == ".md" and path.is_file():
            table_collection.add_table(
                Table.from_markdown(path.path)
            )


def get_repos(target_user: NamedUser.NamedUser, exclusion_repo_full_names: set) -> RepoCollection:
    all_repos = []

    for repo in target_user.get_repos(type="all"):
        if repo.private:
            continue

        if repo.full_name in exclusion_repo_full_names:
            continue

        repo_model = Repository.from_model(repo)

        if repo_model.is_filter:
            continue

        all_repos.append(repo_model)

    repo_collection = RepoCollection(all_repos)

    return repo_collection


def render_readme(tables: TableCollection, template_name: str, out_file: str):
    env = Environment(
        loader=FileSystemLoader("templates"),
        autoescape=select_autoescape(),
        lstrip_blocks=True,
        trim_blocks=True)

    template = env.get_template(template_name)

    with open(out_file, "w", encoding="utf-8") as out:
        for line in template.generate(table_collection=tables):
            out.write(line)


def main(args):
    client = Github(args.gh_token)
    target_user = client.get_user(args.gh_user)

    categories = load_categories(args.cat_list)
    exclusion_repo_list = load_exclusion_repo_list(args.repo_filter_file)

    table_collection = TableCollection()
    load_static_tables(table_collection, args.md_static_data)

    repos = get_repos(target_user, exclusion_repo_list)

    for repo in tqdm(repos.all_repos, mininterval=3):
        for table in Table.from_repo(repo, categories):
            table_collection.add_table(table)

    render_readme(table_collection, args.template_name, args.out_file)


if __name__ == "__main__":
    parser = configargparse.ArgumentParser()
    parser.add_argument("--gh_token", dest="gh_token", type=str,
                        help="A GitHub REST API token", env_var="GITHUB_TOKEN")
    parser.add_argument("--gh_user", dest="gh_user", default="KernelA", type=str,
                        help="A GitHub user name", env_var="GITHUB_TARGET_USER")
    parser.add_argument("--md_static_data", dest="md_static_data", default="./static_data", type=str, required=False,
                        help="A path to static data")
    parser.add_argument("--cat_list", dest="cat_list", default="./categories/cat.json", type=str, required=False,
                        help="A path to json with categories info")
    parser.add_argument("--template_name", dest="template_name", default="my_projects.md.j2", type=str, required=False,
                        help="A path to main Jinja template file")
    parser.add_argument("--out_file", dest="out_file", default="./pages/my_projects.md", type=str, required=False,
                        help="A path to rendered file")
    parser.add_argument("--repo_filter_file", dest="repo_filter_file", default="./exclude_repos/exclusion_list.json", type=str, required=False,
                        help="A path to rendered file")

    args = parser.parse_args()

    main(args)
