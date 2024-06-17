import numpy as np
from sklearn.cluster import KMeans

# 1. Movielens 1M 데이터셋의 "ratings.dat" 파일만 사용
file = 'ratings.dat'

# 2. "ratings.dat"파일을 기반으로 6040 x 3952 크기의 Matrix 생성
ratings = np.empty((6040, 3952), dtype=np.int64)
for line in open(file, 'r'):
    datas = line.split('::')

    i = int(datas[0])
    j = int(datas[1])
    k = int(datas[2])

    ratings[i - 1, j - 1] = k

# 3. User Vector(Matrix의 각 Row)를 기반으로 KMeans 클러스터링을 수행하여 3개의 그룹을 생성
#  - Null값의 경우, 0으로 채워 클러스터링 수행
km = KMeans(n_clusters=3, random_state=0)
km.fit(ratings)

ratings_cluster = [np.where(km.labels_ == i) for i in range(3)]

# 4-1. 그룹 추천 알고리즘

# Additive Utilitarian: 상품에 대한 모든 사용자 평점의 합
def group_recommend_au(idx):
    result = ratings[ratings_cluster[idx]].sum(axis=0)
    return np.argsort(result)[:10][::-1]

# Average: 상품에 대한 모든 사용자 평점의 평균
def group_recommend_avg(idx):
    result = ratings[ratings_cluster[idx]].mean(axis=0)
    return np.argsort(result)[:10][::-1]

# Simple Count: 상품에 평점을 매긴 사용자 수
def group_recommend_sc(idx):
    result = np.count_nonzero(ratings[ratings_cluster[idx]], axis=0)
    return np.argsort(result)[:10][::-1]

# Approval Voting: Simple Count 중 4점 이상의 평점 수
def group_recommend_av(idx):
    # (ratings >= 4) : [ [ True False True, ... ], [ ... ], ... ]
    result = (ratings[ratings_cluster[idx]] >= 4).sum(axis=0)
    return np.argsort(result)[:10][::-1]

# Borda Count: 사용자들의 상품에 대한 랭킹 점수의 합
def group_recommend_bc(idx):
    result = np.argsort(np.argsort(-ratings[ratings_cluster[idx]], axis=1))
    return np.argsort(result.sum(axis=0))[:10][::-1]

# Copeland Rule: 상품 i와 j를 비교하여 i의 평점이 높은 사용자가 많으면 +1 (j는 -1), i의 평점이 높은 사용자와 j의 평점이 높은 사용자의 수가 같으면 0, i를 제외한 모든 상품과 비교하여 총합을 구함
def group_recommend_cr(idx):
    ratings_tmp = ratings[ratings_cluster[idx]]
    rows, columns = ratings_tmp.shape
    result = np.zeros(columns)
    for i in range(rows):
        for j in range(columns):
            if i == j:
                continue
            result[i] += (ratings_tmp[:, i] > ratings_tmp[:, j]).sum()
    return np.argsort(result)[:6][::-1]

# 4. 3개의 그룹 별로 6개의 그룹 추천 알고리즘을 통해 상위 10개의 상품을 찾음 (3 x 6 = 18개의 top 10 결과 출력)
for i in range(3):
    print(f"\nGroup. {i}")
    print(f"Average: {group_recommend_avg(i)}")
    print(f"Additive Utilitarian: {group_recommend_au(i)}")
    print(f"Simple Count: {group_recommend_sc(i)}")
    print(f"Approval Voting: {group_recommend_av(i)}")
    print(f"Borda Count: {group_recommend_bc(i)}")
    print(f"Copeland Rule: {group_recommend_cr(i)}")