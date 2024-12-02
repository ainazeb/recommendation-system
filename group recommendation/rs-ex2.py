import warnings
import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)  # skip pandas Future Warnings
warnings.simplefilter(action='ignore', category=RuntimeWarning)  # skip pandas Runtime Warning

# ---

data = pd.read_csv('ml-latest-small/ratings.csv')


data = data[:5000]  # Limit Data for fast run


# ----
# Q1
# ----


def group_pred_by_avg(group_rates, i):
    rates = group_rates.where(lambda x: x.movieId == i).dropna()
    if rates.empty:
        return 0

    return rates.rating.mean()


def group_pred_by_least_misery(group_rates, i):
    rates = group_rates.where(lambda x: x.movieId == i).dropna()
    if rates.empty:
        return 0

    return rates.rating.min()


group_users = [58, 35, 2]
rates_of_group = []
for user in group_users:
    rates_of_group.append(data.where(lambda x: x.userId == user).dropna())

rates_of_group = pd.concat(rates_of_group)

print(f"A. Top 20 Recommended Movies for group:{group_users}")
print(f"Part 1. By Average Aggregation:")

recommended = []
for movieId in data.movieId.unique():
    recommended.append({
        'movieId': movieId,
        'rate': group_pred_by_avg(rates_of_group, movieId)
    })

recommended = sorted(recommended, key=lambda i: i['rate'], reverse=True)  # sort

idx = 1
for each in recommended:
    print(f"{idx}. movieId:{each['movieId']} with pred:{each['rate']}")
    idx += 1

    if idx > 20:
        break


print(f"\n Part 2. By Least Misery Aggregation:")

recommended = []
for movieId in data.movieId.unique():
    recommended.append({
        'movieId': movieId,
        'rate': group_pred_by_least_misery(rates_of_group, movieId)
    })

recommended = sorted(recommended, key=lambda i: i['rate'], reverse=True)  # sort

idx = 1
for each in recommended:
    print(f"{idx}. movieId:{each['movieId']} with pred:{each['rate']}")
    idx += 1

    if idx > 20:
        break


# ----
# Q2
# ----


def group_pred_with_disagreements(group_rates, i):
    rates = group_rates.where(lambda x: x.movieId == i).dropna()
    if rates.empty:
        return 0

    scores = [x - 2.5 for x in rates.rating]

    return np.array(scores).mean()


print(f"\n\n B. Top 20 Recommended Movies for group:{group_users} with disagreements")
recommended = []
for movieId in data.movieId.unique():
    recommended.append({
        'movieId': movieId,
        'rate': group_pred_with_disagreements(rates_of_group, movieId)
    })

recommended = sorted(recommended, key=lambda i: i['rate'], reverse=True)  # sort

idx = 1
for each in recommended:
    print(f"{idx}. movieId:{each['movieId']} with pred:{each['rate']}")
    idx += 1

    if idx > 20:
        break
