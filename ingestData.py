import csv
from collections import defaultdict
import datetime
import numpy as np
from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score

dataPath = "./BusinessAnalytics/"

gameDataFile = dataPath + "game_data.csv"
playerDataFile = dataPath + "player_data.csv"
trainingFile = dataPath + "training_set.csv"
testFile = dataPath + "test_set.csv"
outFile = "predicted_viewership.csv"

# Formula we're graded on, from the pdf
def MAPE(predicted_responses, actual_responses):
    n = len(predicted_responses)
    currTotal = 0
    for i in range(n):
        ai = actual_responses[i]
        pi = predicted_responses[i]
        numerator = ai - pi
        denominator = ai
        currTotal += np.absolute(float(numerator/denominator))
    return float(currTotal / n)


teams = set()
# Number of samples (games) in training set
numDataPoints = 0
# Number of samples (games) in test set
numTestDataPoints = 0

# For ignoring test data while training
testSetGameIds = set()
with open(testFile, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        testSetGameIds.add(row['Game_ID'])

# Maps game ID to the total number of viewers across all countries
# To improve, try for each country independently (is there enough data
# for that to be useful?)
viewersPerGame = defaultdict(int)    
with open (trainingFile, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        gameId = row['Game_ID']            
        viewersPerGame[gameId] += int(row['Rounded Viewers'])

# Dictionary of information for each game
# {gameID: {'H':{homeTeamInfo}, 'A':{awayTeamInfo}}}
gameInfoDict = {}
testGameInfoDict = {}
with open(gameDataFile, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        teams.add(row['Team'])
        gameId = row['Game_ID']

        # Only gathering training currently
        if gameId in testSetGameIds:
            if gameId in testGameInfoDict:
                currentData = testGameInfoDict[gameId]
            else:
                currentData = {}
                numTestDataPoints += 1
        else:
            if gameId in gameInfoDict:
                currentData = gameInfoDict[gameId]
            else:
                currentData = {}
                numDataPoints += 1
        
        # loc can be 'H' or 'A'
        loc = row['Location']
        currentData[loc] = {}

        currentData[loc]['wins'] = int(row['Wins_Entering_Gm'])
        currentData[loc]['losses'] = int(row['Losses_Entering_Gm'])
        currentData[loc]['team'] = row['Team']

        if gameId in testSetGameIds:
            testGameInfoDict[gameId] = currentData
        else:
            gameInfoDict[gameId] = currentData



### ABOVE ARE DEFINITIONS, READING IN INITIAL DATA ###
### BELOW IS SETTING UP, TRAINING, AND TESTING THE MODEL ###

teamsSorted = sorted(list(teams))
numTeams = len(teams)
# Column plan: sorted teams for binary 1/0, followed by home team win, home team loss, away team win, away team loss
numVariables = numTeams + 4 
homeWinIdx = numTeams
awayWinIdx = numTeams + 1
homeLossIdx = numTeams + 2
awayLossIdx = numTeams + 3


### TRAINING ###

# Training matrices initialization
trainingData = np.zeros((numDataPoints, numVariables))
trainingShape = trainingData.shape
print("Number of training samples: {}\nNumber of variables: {}".format(trainingShape[0],trainingShape[1]))
trainingValues = np.zeros((numDataPoints, 1))

dataPoint = 0

for gameId in gameInfoDict:
    game = gameInfoDict[gameId]
    gameInfoHome = game['H']
    gameInfoAway = game['A']
    teamIndexHome = teamsSorted.index(gameInfoHome['team'])
    trainingData[dataPoint][teamIndexHome] = 1
    teamIndexAway = teamsSorted.index(gameInfoAway['team'])
    trainingData[dataPoint][teamIndexAway] = 1
    trainingData[dataPoint][homeWinIdx] = gameInfoHome['wins']
    trainingData[dataPoint][awayWinIdx] = gameInfoAway['wins']
    trainingData[dataPoint][homeLossIdx] = gameInfoHome['losses']
    trainingData[dataPoint][awayLossIdx] = gameInfoAway['losses']
    trainingValues[dataPoint][0] = viewersPerGame[gameId]
    dataPoint += 1

if dataPoint != trainingShape[0]:
    print("Number of data points found: {}".format(dataPoint))

# Runs the regression
ourModel = linear_model.LinearRegression()
ourModel.fit(trainingData, trainingValues)

# Compares the trained scores on the original set to their actual values
# Using r^2, not MAPE
print("Score on training set: {}".format(ourModel.score(trainingData, trainingValues)))

resultsOnTrainingSet = ourModel.predict(trainingData)
print("MAPE on training set: {}".format(MAPE(resultsOnTrainingSet.flatten().tolist(), trainingValues.flatten().tolist())))


### TESTING ###
# I know this is super ugly and duplicative but we only have two
# files/sets and there were too many moving parts, not enough time

# Test matrices initialization
testData = np.zeros((numTestDataPoints, numVariables))
testShape = testData.shape
print("Number of test samples: {}".format(testShape[0]))

dataPoint = 0
# For keeping track of test entries in the dictionary
testGameIdxDict = {}

for gameId in testGameInfoDict:
    testGameIdxDict[gameId] = dataPoint
    game = testGameInfoDict[gameId]
    gameInfoHome = game['H']
    gameInfoAway = game['A']
    teamIndexHome = teamsSorted.index(gameInfoHome['team'])
    testData[dataPoint][teamIndexHome] = 1
    teamIndexAway = teamsSorted.index(gameInfoAway['team'])
    testData[dataPoint][teamIndexAway] = 1
    testData[dataPoint][homeWinIdx] = gameInfoHome['wins']
    testData[dataPoint][awayWinIdx] = gameInfoAway['wins']
    testData[dataPoint][homeLossIdx] = gameInfoHome['losses']
    testData[dataPoint][awayLossIdx] = gameInfoAway['losses']
    dataPoint += 1

if dataPoint != testShape[0]:
    print("Number of data points found: {}".format(dataPoint))


resultsOnTestSet = ourModel.predict(testData)

with open(testFile, 'r') as testF:
    reader = csv.DictReader(testF)
    headers = reader.fieldnames
    with open(outFile, 'w') as outF:
        writer = csv.DictWriter(outF,fieldnames=headers)
        writer.writeheader()
        dataPoint = 0
        for line in reader:
            gameId = line['Game_ID']
            gameIdx = testGameIdxDict[gameId]
            line['Total_Viewers'] = resultsOnTestSet[dataPoint][0]
            writer.writerow(line)
            dataPoint += 1





# to be printed to file

# This is where we should add new features (wut about win % instead of two features?)
# Also an option! This was just the most direct option
# I think we should also so differential between the two teams
# Less viewership if its warriors vs suns
# Let me know if you have questions, I think the rest is self explanatory? hard to tell since i wrote it ahha
# Also youre totally right, percentage makes way more sense, except... 
# well, idk, how do we account for that the earlier games will be so spread out, like 0 or 100% in the beginning isnt accurate
# maybe literally add date (in a sortable format, which i dont think it is right now)
# yes
# okay
# yeah maybe percentage doesn't matter, just totals and difference
# i honestly am not positive either way, so lets try all the options and whatever has a better MAPE on the training set wins?
# is sam getting on tonight?
# I want to go get sushi
# not sure how those are related
# but i think yes
# ah its so easy
# fine ill do it
# I will try to use lineup data/google trends though for a feature
# cool! yeah feel free to do that then, maybe it makes more sense for me to continue with all of the more basic stuff since i started it
#although i'm happy to brainstorm more features. I'm gonna go get sushi first though
# gotcha, sounds good

# # player stuff, not worrying about this yet
# maxScore = defaultdict(int)
# with open(playerDataFile, 'r') as f:
#     reader = csv.DictReader(f)
#     for row in reader:
#         player = row['Person_ID']
#         if row['Active_Status']:
#             if row['Points'] > maxScore[player]:
#                 maxScore[player] = row['Points']



# # ROUGH SKETCH!
# gameId: {
#     homeTeamWins: int`home_wins_entering`,
#     awayTeamWins: int`away_wins_entering`,
#     homeTeamLosses: int`whatever`,
#     awayTeamLosses: int`you get it`,
#     CLE: 0/1
#     NYK: 0/1
#     POR: 0/1 #a bool value for all thirty-something teams in one game vector thing? | fair
# #    etc. with binary representing whether or not that team plays the game
# # yeah except not a boolian, literally the number 0 or the number 1
# # that's generally how you treat categorical data with this kind of ml
# # either it belongs in the category or it doesnt
# }