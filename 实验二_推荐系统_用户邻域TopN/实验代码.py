from __future__ import annotations

import argparse
import math
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "数据集"
DATASET_DIR = DATA_DIR / "ml-100k"
ZIP_PATH = DATA_DIR / "ml-100k.zip"
DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"


def ensure_dataset() -> None:
    """如果本地没有 MovieLens 100K 数据集，则自动下载并解压。"""
    required_files = [DATASET_DIR / "u.data", DATASET_DIR / "u.item"]
    if all(path.exists() for path in required_files):
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("未检测到数据集，开始下载 MovieLens 100K...")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)

    print("下载完成，开始解压...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:
        zip_file.extractall(DATA_DIR)
    print(f"数据集已准备完成：{DATASET_DIR}")


def load_movie_titles() -> dict[int, str]:
    movie_titles: dict[int, str] = {}
    item_path = DATASET_DIR / "u.item"
    with item_path.open("r", encoding="latin-1") as file:
        for line in file:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                movie_id = int(parts[0])
                movie_titles[movie_id] = parts[1]
    return movie_titles


def load_positive_interactions(min_rating: int) -> dict[int, list[tuple[int, int]]]:
    """读取评分数据，只保留高于阈值的正反馈。"""
    interactions: dict[int, list[tuple[int, int]]] = defaultdict(list)
    rating_path = DATASET_DIR / "u.data"
    with rating_path.open("r", encoding="utf-8") as file:
        for line in file:
            user_id_str, movie_id_str, rating_str, timestamp_str = line.strip().split("\t")
            rating = int(rating_str)
            if rating >= min_rating:
                user_id = int(user_id_str)
                movie_id = int(movie_id_str)
                timestamp = int(timestamp_str)
                interactions[user_id].append((movie_id, timestamp))

    # 按时间排序，方便留一法划分训练集和测试集
    for user_id in interactions:
        interactions[user_id].sort(key=lambda row: row[1])
    return interactions


def train_test_split(
    interactions: dict[int, list[tuple[int, int]]]
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """对每个用户采用留一法：最后一次正反馈放入测试集。"""
    train: dict[int, set[int]] = {}
    test: dict[int, set[int]] = {}

    for user_id, records in interactions.items():
        items = [movie_id for movie_id, _ in records]
        unique_items = list(dict.fromkeys(items))
        if len(unique_items) < 2:
            train[user_id] = set(unique_items)
            continue

        test[user_id] = {unique_items[-1]}
        train[user_id] = set(unique_items[:-1])

    return train, test


def build_user_similarity(train: dict[int, set[int]]) -> dict[int, dict[int, float]]:
    """基于共同喜欢的物品计算用户相似度。"""
    item_users: dict[int, set[int]] = defaultdict(set)
    for user_id, items in train.items():
        for item_id in items:
            item_users[item_id].add(user_id)

    co_rated: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    user_item_count: dict[int, int] = defaultdict(int)

    for item_id, users in item_users.items():
        user_list = list(users)
        if not user_list:
            continue
        weight = 1.0 / math.log(1.0 + len(user_list))
        for user_id in user_list:
            user_item_count[user_id] += 1
        for i, user_u in enumerate(user_list):
            for user_v in user_list[i + 1 :]:
                co_rated[user_u][user_v] += weight
                co_rated[user_v][user_u] += weight

    similarity: dict[int, dict[int, float]] = defaultdict(dict)
    for user_u, related_users in co_rated.items():
        for user_v, cuv in related_users.items():
            similarity[user_u][user_v] = cuv / math.sqrt(
                user_item_count[user_u] * user_item_count[user_v]
            )
    return similarity


def recommend(
    user_id: int,
    train: dict[int, set[int]],
    similarity: dict[int, dict[int, float]],
    top_k_neighbors: int,
    top_n_items: int,
) -> list[tuple[int, float]]:
    watched_items = train.get(user_id, set())
    rank: dict[int, float] = defaultdict(float)

    neighbors = sorted(
        similarity.get(user_id, {}).items(), key=lambda row: row[1], reverse=True
    )[:top_k_neighbors]

    for neighbor_id, sim_score in neighbors:
        for item_id in train.get(neighbor_id, set()):
            if item_id in watched_items:
                continue
            rank[item_id] += sim_score

    return sorted(rank.items(), key=lambda row: row[1], reverse=True)[:top_n_items]


def evaluate(
    train: dict[int, set[int]],
    test: dict[int, set[int]],
    similarity: dict[int, dict[int, float]],
    top_k_neighbors: int,
    top_n_items: int,
) -> dict[str, float]:
    hit = 0
    total_recommend = 0
    total_test = 0
    all_recommended_items: set[int] = set()

    item_popularity: dict[int, int] = defaultdict(int)
    for items in train.values():
        for item_id in items:
            item_popularity[item_id] += 1

    popularity_sum = 0.0
    evaluated_users = 0

    for user_id, test_items in test.items():
        if user_id not in train or user_id not in similarity:
            continue

        rec_items = recommend(user_id, train, similarity, top_k_neighbors, top_n_items)
        if not rec_items:
            continue

        evaluated_users += 1
        recommended_ids = {item_id for item_id, _ in rec_items}
        hit += len(recommended_ids & test_items)
        total_recommend += len(rec_items)
        total_test += len(test_items)
        all_recommended_items.update(recommended_ids)

        for item_id, _ in rec_items:
            popularity_sum += math.log(1 + item_popularity[item_id])

    catalog = {item_id for items in train.values() for item_id in items}

    precision = hit / total_recommend if total_recommend else 0.0
    recall = hit / total_test if total_test else 0.0
    coverage = len(all_recommended_items) / len(catalog) if catalog else 0.0
    popularity = popularity_sum / total_recommend if total_recommend else 0.0

    return {
        "evaluated_users": evaluated_users,
        "precision": precision,
        "recall": recall,
        "coverage": coverage,
        "popularity": popularity,
    }


def print_sample_recommendations(
    user_id: int,
    train: dict[int, set[int]],
    similarity: dict[int, dict[int, float]],
    movie_titles: dict[int, str],
    top_k_neighbors: int,
    top_n_items: int,
) -> None:
    print(f"\n为用户 {user_id} 生成 Top-{top_n_items} 推荐结果：")
    results = recommend(user_id, train, similarity, top_k_neighbors, top_n_items)
    if not results:
        print("该用户暂无可推荐结果。")
        return

    for index, (item_id, score) in enumerate(results, start=1):
        movie_title = movie_titles.get(item_id, f"Movie {item_id}")
        print(f"{index:>2}. {movie_title:<50} 相似度加权得分: {score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="基于用户邻域的 Top-N 推荐实验")
    parser.add_argument("--min-rating", type=int, default=4, help="正反馈评分阈值，默认 4")
    parser.add_argument("--topk", type=int, default=20, help="邻居用户数量，默认 20")
    parser.add_argument("--topn", type=int, default=10, help="推荐物品数量，默认 10")
    parser.add_argument("--sample-user", type=int, default=1, help="展示推荐结果的示例用户")
    args = parser.parse_args()

    ensure_dataset()
    movie_titles = load_movie_titles()
    interactions = load_positive_interactions(args.min_rating)
    train, test = train_test_split(interactions)
    similarity = build_user_similarity(train)
    metrics = evaluate(train, test, similarity, args.topk, args.topn)

    print("\n========== 基于用户邻域的 Top-N 推荐实验 ==========")
    print(f"数据集: MovieLens 100K")
    print(f"正反馈阈值: rating >= {args.min_rating}")
    print(f"邻居数 K: {args.topk}")
    print(f"推荐列表长度 N: {args.topn}")
    print(f"训练用户数: {len(train)}")
    print(f"测试用户数: {len(test)}")
    print(f"参与评估用户数: {int(metrics['evaluated_users'])}")
    print(f"Precision@{args.topn}: {metrics['precision']:.4f}")
    print(f"Recall@{args.topn}: {metrics['recall']:.4f}")
    print(f"Coverage@{args.topn}: {metrics['coverage']:.4f}")
    print(f"Popularity@{args.topn}: {metrics['popularity']:.4f}")

    if args.sample_user in train:
        print_sample_recommendations(
            args.sample_user,
            train,
            similarity,
            movie_titles,
            args.topk,
            args.topn,
        )
    else:
        print(f"\n用户 {args.sample_user} 不在训练集中，请更换 sample-user。")


if __name__ == "__main__":
    main()
