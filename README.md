# rental-product-recommendation
1.Project Overview
Source:https://www.kaggle.com/competitions/rental-product-recommendation-system/overview
A Kaggle Competition
In one Word:Two-stage session-based recommender
Data source:Yandex Metrica
Goal:Top-6 item recommendation per session

2.Problem Formulation
Input:Session behaviour sequence
Output:Top-6 commodity code
Evaluation indicators:Recall@6 / NDCG@6(Offline),Kaggle Public Score(Online)

3.System Architecture
User Session
   ↓
ItemCF Retrieval (Top-N)
   ↓
Candidate Union + Cold-start Fallback
   ↓
LightGBM LambdaRank (NDCG@6)
   ↓
Top-6 Recommendation

4.Modeling Details
Recall: Item-based CF with co-visitation + IUF weighting
Cold-start: traffic_source / device-level popularity buckets
Ranking: LightGBM LambdaRank (grouped by session)
Features: CF score, candidate rank, session length, order-based popularity, context features

5.Results
| Stage        | Metric           | Result    |
| ------------ | ---------------- | --------- |
| ItemCF only  | Public Score     | ~0.11     |
| + Cold-start | Public Score     | ~0.15     |
| + LambdaRank | Offline Recall@6 | 0.64      |
| Final        | Public Score     | **0.23+** |

6.Notes on Data Leakage
During experimentation, severe train–test identifier overlap was observed in the dataset.
To ensure a production-oriented solution, all forms of label leakage were deliberately avoided.

