# dfs

# Steps to Run Daily
1. Go to BigDataBall and download the most recent in-season data. Move to data folder and delete the .csv version
2. Go to DraftKings and download the salary data template for the slate you want
3. Go to RotoGrinders and download the daily projections for the correct slate

TODO:
-- 1. Fix evaluate function to avoid double counting where multiple entries hit target
-- 2. Implement simulated annealing
-- 3. Test GA against SA and try a hybrid approach. Record results
-- 4. Try different params for lineup pool generation
-- 5. Look at results of selected lineups with far more simulations/different seed. 50k sims probably isn't enough to get
   a good sense of upper tail behaviours. Runtime might be issue with too many sims.
6. Think about the size of the lineup pool. E.g. order of magnitude for "plausible" lineups, i.e. lineups that have a shot
   at winning. Might need to up size of lineup pool
       - Run a test with far more parameters, if it works well, can use more computing power


- Look at probability of winning as a function of the number of lineups. Can use multiple accounts...
- Redefine "winning" as hitting a moving target that's adjusted for number of entrants, number of games, average player valuesG