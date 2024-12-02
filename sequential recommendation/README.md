This Python notebook demonstrates a group recommendation system for movies using a Sequential Hybrid Aggregation Model that combines average aggregation, least misery, and Borda count methods. The project leverages the MovieLens dataset for user ratings and includes functionalities for both user-based and group-based recommendation systems.

Key Features:
Dataset Processing: Downloads and prepares the MovieLens dataset for analysis.
User-Based Recommendations: Computes personalized movie predictions using similarity-based collaborative filtering.
Group-Based Recommendations: Aggregates preferences for groups using hybrid scoring:
Average Aggregation
Least Misery Aggregation
Borda Count Aggregation
Satisfaction and Disagreement Metrics: Measures user satisfaction, group satisfaction, and disagreement to refine recommendations iteratively.
Sequential Hybrid Model: Dynamically adjusts weights for aggregations (alpha, beta, gamma) based on group satisfaction and variance to balance individual and group preferences.