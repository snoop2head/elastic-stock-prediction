### Adding more columns to stock data

- [x] Include nasdaq and S&P500, etc.
- [x] [Implement ARIMA and RandomForestRegressor from indigo team's implementation](https://dacon.io/competitions/official/235800/codeshare/3320?page=1&dtype=recent)
- [x] ~~채권, 금리, 실업 등 거시경제 요소 데이터 반영하기.~~ -> Performance got worse
- [x] Increase the number of iterations for elasticnetCV
- [x] **Make elasticnet to include the recent days**
- [x] Fix dropna() and .iloc[:-5] double slicing problem
- [x] Update public mae for the new weekdays
  - [x] Sampled ElasticNetCV shows 3.0 for the public leaderboard period.
  - [x] Sampled ElasticNetCV shows 2.97 for the new public period!
  - [x] ffill ElasticNetCV for new_public shows 2.93
  - [x] ffill ElasticNetCV for public period shows 3.70
- [x] Autoarima for p, d, q optimization -> AutoArima yields 8.8
  - [x] 1. based on new loss function -> mae
  - [x] 2. apply different p, d, q according to 376 different companies

---

- [ ] ~~To resolve overfitting to the public leaderboard period, make another loss function for mape() on longer & recent period than public period. mape() currently checks for only 5 days -> change it to 15 days within a month~~ -> replaced with cross validation
- [ ] ~~Find the right PHLX data~~
- [ ] ~~SHIFT the WTI and PHLX data +1 or -1 day~~
- [ ] ~~Fill the WTI and PHLX data for missing holiday dates~~
- [ ] ~~Add 1, 2, 3, 4, 5 ... 3 months of data rather than individual date data~~
- [ ] ~~**Quantking 재무 데이터 가져오기(~2019년 or ~ 2016년)**~~
- [ ] ~~**ffill, bfill -> 이동평균선이 더 나을 수도 있음. rolling을 쓸까?**~~

### Modeling

- [x] Apply Cross Validation for sklearn's linear model ElasticNet
  - [x] [Used ElasticCV model for optimizing hyperparameter alpha](./main_ver2_ElasticNetCV.ipynb)
- [x] Apply ARIMA
- [x] Ensemble ARIMA and ElasticNetCV
- [x] check data leakage for randomforest regressor
- [x] Visualize model's prediction value.
- [ ] find the proper ensemble ratio for the ensemble and record it on the csv file
- [ ] ~~Apply Cross validation for RandomForestRegressor~~
- [ ] ~~Apply grid search for RandomForestRegressor~~
- [ ] ~~Check the validity of RandomForestRegressor's private prediction~~
- [ ] ~~Apply DNN -> only predicts linear seasonality~~
