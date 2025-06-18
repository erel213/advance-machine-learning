Description
This challenge uses a curated version from the M5 dataset to test your ability to create accurate and innovative forecasting projections. The dataset includes daily revenue from 10 Walmart stores.

M5 is the fifth M competition, named after Professor Spyros Makridakis from the University of Nicosia, Cyprus.
https://www.unic.ac.cy/iff/research/forecasting/
https://en.wikipedia.org/wiki/Makridakis_Competitions

Dataset Details
Training Data: Contains 18,766 data points for model training.
Forecasting Submission: 1,012 entries are required to predict for submission.
Key Features
Calendar data with information about events that can affect retail sales.
Objectives
Build a hierarchical forecasting model that can predict the store revenue for different hierarchies (per store, all stores).
Leverage exogenous data to enhance your model.
Explore different models, explainability, and innovative approaches to improve forecasting quality.
Whether you’re experimenting with statistical models, machine learning, deep learning, or large pretrained models, this competition offers opportunities to demonstrate and refine your skills while pushing the boundaries of real data forecasting.

Evaluation
Submissions are evaluated with RMSE between the predicted value and the observed value.


Note that stores with a higher revenue baseline will probably have a greater impact on the RMSE. Additionally, the evaluation of all stores, by design, will carry more weight in this approach.

Submission File
You must forecast a value for each user store ID and date of the submission file. The file should contain a header and have the following format:

id, prediction
0_20250101, 100
0_20250102, 200
0_20250103, 100
0_20250104, 100
...
Where id is a concatenation of {store_id}_{date_formatted}
date formatted is a string format of a date without separators (e.g., 2025-12-31 --> 20251231)

Grading
Participants must meet the following milestones to score points:
Requirements (60 points)
Notebook Submission: Submit your notebook through Moodle.
Achieve Baseline: An RMSE lower than the baseline on both the Public and Private Leaderboards (e.g., RMSE of 11,761 or lower on the Public Leaderboard).
Note: The Requirements section is mandatory; you either get all points or none of them.

Competition Score (0 - 40 points)
Your final score in the competition reflects how much you improve over a simple baseline model and how your performance compares to the best-performing group.

We use the following formula:


BaselineRMSE: The RMSE score of the baseline model from the Private Leaderboard.
GroupRMSE: The RMSE score of your model from the Private Leaderboard.
BestRMSE: The RMSE score achieved by the best group (ranked 1) from the Private Leaderboard.
What this means:
A perfect score of 40 goes to the group with the best RMSE (totaling 100 points).
A model with the same RMSE as the baseline will not receive an additional score (totaling 60 points).
Groups receive scores proportional to their improvement, relative to the baseline and the best group's performance. This approach ensures that even small differences in performance near the top are rewarded fairly, while still encouraging everyone to beat the baseline.

Course Bonus
The top 5 ranking groups will be invited to give a short presentation of their work. Participation is required to receive the bonus.
Each presenting group will be eligible for up to 5 bonus points toward their overall course grade, based on the quality of their presentation.
If a group chooses not to present, the opportunity will be offered to the next-highest ranked group (e.g., 6th place and so on).
This is a great opportunity to showcase your work and contribute to the learning of your classmates.