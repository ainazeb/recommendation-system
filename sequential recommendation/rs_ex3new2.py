# -*- coding: utf-8 -*-
"""rs-ex3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qQevL32m28sSXRs8fX6GLDYSLwXmAj9k

# Download Dataset
"""

!wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
!unzip ml-latest-small.zip

"""# Import Libraries"""

import math
import warnings
import numpy as np
import pandas as pd

warnings.simplefilter(action='ignore', category=FutureWarning)  # skip Future Warnings for a more clean log
warnings.simplefilter(action='ignore', category=RuntimeWarning)  # skip Runtime Warnings for a more clean log

"""# Load Datasets"""

data = pd.read_csv('./ml-latest-small/ratings.csv')

data = [
    data[:10000], #subset 1
    data[:20000], #subsets 1-2
    data[:30000], #subsets 1-3
    data[:40000], #subsets 1-4
    data[:50000], #subsets 1-5
    data[:60000], #subsets 1-6
    data[:70000], #subsets 1-7
    data[:80000], #subsets 1-8
    data[:90000], #subsets 1-9
    data, #subsets 1-10
]

"""# User Based Recommendation Systems"""

def user_based_sim(rates_by_a, rates_by_b):
    rate_avg_a = rates_by_a.rating.mean()
    rate_avg_b = rates_by_b.rating.mean()

    similar_rated_movies = list(np.intersect1d(rates_by_a.movieId, rates_by_b.movieId))

    if len(similar_rated_movies) == 0:
        return 0

    fraction_top = 0
    fraction_bottom_a = 0
    fraction_bottom_b = 0

    for mid in similar_rated_movies:
        rate_of_a = rates_by_a[rates_by_a.movieId == mid].rating.mean()
        rate_of_b = rates_by_b[rates_by_b.movieId == mid].rating.mean()

        fraction_top += (rate_of_a - rate_avg_a) * (rate_of_b - rate_avg_b)
        fraction_bottom_a += (rate_of_a - rate_avg_a) ** 2
        fraction_bottom_b += (rate_of_b - rate_avg_b) ** 2

    
    frac = fraction_top / (math.sqrt(fraction_bottom_a) * math.sqrt(fraction_bottom_b))
    return 1 if np.isnan(frac) else frac


def user_based_pred(iteration_data, rates_by_a, p):
    if rates_by_a.empty:
      return 0
    
    rate_of_a = rates_by_a[rates_by_a.movieId == p].rating
    if not rate_of_a.empty: # if already rated to this movie use his own rate
      return rate_of_a.mean()

    rate_avg_a = rates_by_a.rating.mean()

    N = iteration_data[iteration_data.movieId == p].userId

    fraction_top = 0
    fraction_bottom = 0

    for b in N:
        rates_by_b = iteration_data[iteration_data.userId == b]
        rate_avg_b = rates_by_b.rating.mean()
        rate_of_b = rates_by_b[rates_by_b.movieId == p].rating.mean()
        
        similarity = user_based_sim(rates_by_a, rates_by_b)
        fraction_top += similarity * (rate_of_b - rate_avg_b)
        fraction_bottom += abs(similarity)

    try:
        frac = fraction_top / fraction_bottom
        frac = 0 if np.isnan(frac) else frac
    except ZeroDivisionError:
        frac = 0
    
    return min([5, rate_avg_a + frac])

"""# Group Based Recommendation Systems"""

def group_pred_by_3_ways(iteration_data, group_rates, groupUserIds, i):
    rates = []

    for userId in groupUserIds:
      pred = user_based_pred(iteration_data, group_rates[group_rates.userId == userId], i)
      rates.append(pred)

    avg_agg = sum(rates) / len(rates)
    least_misery_agg = min(rates)

    movieRateSummation = {}
    for movieId in iteration_data.movieId.unique():
      movieRateSummation[movieId] = np.sum(group_rates[group_rates.movieId == movieId].rating)

    movieRateSummation[i] += sum(rates)
    movieRateSummation = [{"id": movieId, "rate": movieRate} for movieId, movieRate in movieRateSummation.items()]
    np_rates = [x['id'] for x in sorted(movieRateSummation, key=lambda x: x['rate'], reverse=True)]
    movie_idx = np_rates.index(i)
    borda_count_agg = ((len(movieRateSummation) - movie_idx) * 5) / len(movieRateSummation)

    return avg_agg, least_misery_agg, borda_count_agg

def user_sat(iteration_data, predictions, rates_by_group, top_k_user_rates, userId):
  rates_by_user = rates_by_group[rates_by_group.userId == userId]


  group_list_sat = np.sum([user_based_pred(iteration_data, rates_by_user, p) for p in predictions])
  user_list_sat = np.sum(top_k_user_rates.rating)

  try:
    return min([1, group_list_sat / user_list_sat])
  except ZeroDivisionError:
    return 0


def overall_user_sat(satisfactions, userIdx, until_iteration):
    items = []
    for idx in range(1, until_iteration + 1):
      items.append(satisfactions[idx][userIdx])
    
    if len(items) == 0:
      return 0

    return sum(items) / len(items)


def group_sat_iter_j(satisfactions, j):
  return sum(satisfactions[j]) / len(satisfactions[j])


def group_sat(iteration_data, predictions, rates_by_group, groupUserIds):
  sat = sum([user_sat(iteration_data, predictions, rates_by_group, userId, groupUserIds) for userId in groupUserIds])
  return sat / len(groupUserIds)


def overall_group_sat(satisfactions, groupUserIds, until_iteration):
    s = 0
    
    for userIdIndex, userId in enumerate(groupUserIds):
      s += overall_user_sat(satisfactions, userIdIndex, until_iteration)
    
    return s / len(groupUserIds)


def group_disagreement(satisfactions, iteration_j):
    return max(satisfactions[iteration_j]) - min(satisfactions[iteration_j])


def overall_group_disagreement(satisfactions, until_iteration):
    s = 0
    for idx in range(1, until_iteration + 1):
        s += max(satisfactions[idx]) - min(satisfactions[idx])
    
    return s / until_iteration

"""# My New Sequential Hybrid Aggregation Model"""

def sequential_hybrid_agg_model(groupUserIds, j, k):
  prev_user_satisfactions = {0: [1 for userId in groupUserIds]}
  recommended = []
  alpha, beta, gama = 0, 0, 0
  
  for j_idx in range(1, j + 1):
    iteration_data = data[j_idx - 1]
    

    recommended = []

    user_preferences = {}
    rates_by_group = []
    for userId in group_users:
        rates = iteration_data[iteration_data.userId == userId]

        rates_by_group.append(rates)

        user_preferences[userId] = rates.sort_values(by=['rating'], ascending=False)[:k]

    rates_by_group = pd.concat(rates_by_group)
    movieIds = iteration_data.movieId.unique()
    
    sat_avg = sum(prev_user_satisfactions[j_idx - 1]) / len(prev_user_satisfactions[j_idx - 1])
    sat_var = np.var(prev_user_satisfactions[j_idx - 1])

    beta = sat_avg / 2 - sat_var
    gama = sat_avg / 2 + sat_var
    alpha = 1 - (beta + gama)

    for movieId in movieIds:
        avg_aggregation, least_misery_aggregation, borda_aggregation = group_pred_by_3_ways(iteration_data, rates_by_group, groupUserIds, movieId)

        score = alpha * avg_aggregation + beta * borda_aggregation + gama * least_misery_aggregation
        
        recommended.append({
            'movieId': movieId,
            'rate': score,
            'message': f"movieId:{movieId} with pred:{round(score, 3)}, "
                       f"average aggregation:{round(avg_aggregation, 3)}, "
                       f"least misery aggregation:{round(least_misery_aggregation, 3)}, "
                       f"borda aggregation:{round(borda_aggregation, 3)}"
        })

    recommended = sorted(recommended, key=lambda i: i['rate'], reverse=True)[:k] # sort
    predictions = [x['movieId'] for x in recommended]

    
    print(f"Top {k} Recommended Movies for group:{group_users} in iteration {j_idx}")
    for idx, each in enumerate(recommended):
        print(f"{idx + 1}. {each['message']}")

    print("\n")

    # Re-calculate satisfactions
    prev_user_satisfactions[j_idx] = [user_sat(iteration_data, predictions, rates_by_group, user_preferences[userId], userId) for userId in groupUserIds]
    
    # Print Log
    for userIdIndex, userId in enumerate(groupUserIds):
      print(f"userId:{userId} Satisfaction: {round(prev_user_satisfactions[j_idx][userIdIndex] * 100)}%")
      print(f"userId:{userId} Overall Satisfaction: {round(overall_user_sat(prev_user_satisfactions, userIdIndex, j_idx) * 100)}%")
    
    print(f"Group Satisfaction: {round(group_sat_iter_j(prev_user_satisfactions, j_idx) * 100)}%")
    print(f"Group Overall Satisfaction: {round(overall_group_sat(prev_user_satisfactions, groupUserIds, j_idx) * 100)}%")
    print(f"Group Disagreement: {round(group_disagreement(prev_user_satisfactions, j_idx) * 100)}%")
    print(f"Alpha: {alpha}, Beta: {beta}, Gama: {gama}")
    print("\n ---- \n")

"""# TEST"""

# inputs
k = 20
j = 5
group_users = [8, 312, 609]

# run
sequential_hybrid_agg_model(group_users, j, k)