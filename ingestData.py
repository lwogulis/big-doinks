import csv
from collections import defaultdict
import datetime
import numpy as np
import operator
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

def createMatrices(num_samples, num_variables, game_info_dict, viewers_per_game=None):
    # Matrix initialization
    data = np.zeros((num_samples, num_variables))
    values = np.zeros((num_samples, 1))
    dataShape = data.shape
    print("Number of samples: {}".format(dataShape[0]))

    data_point = 0
    # For keeping track of test entries in the dictionary
    # Not used in training
    game_index_dict = {}

    for gameId in game_info_dict:
        game_index_dict[gameId] = data_point
        game = game_info_dict[gameId]
        gameInfoHome = game['H']
        gameInfoAway = game['A']
        teamIndexHome = teamsSorted.index(gameInfoHome['team'])
        data[data_point][teamIndexHome] = 1
        teamIndexAway = teamsSorted.index(gameInfoAway['team'])
        data[data_point][teamIndexAway] = 1
        data[data_point][homeWinIdx] = gameInfoHome['wins']
        data[data_point][awayWinIdx] = gameInfoAway['wins']
        data[data_point][homeLossIdx] = gameInfoHome['losses']
        data[data_point][awayLossIdx] = gameInfoAway['losses']

        if viewers_per_game:
            values[data_point][0] = viewers_per_game[gameId]
        data_point += 1

    if data_point != dataShape[0]:
        print("Unexpected number of data points found: {}".format(data_point))
    
    # For training, need validation matrix
    if viewers_per_game:
        return data_point, data, values
    # For testing, need to be able to refer back to game storage
    else:
        return data_point, data, game_index_dict

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

# ### Gather the top five percent of players for each season 

# # splits player_data into '17 and '18 season
# firstSeasonPlayers = {}
# secondSeasonPlayers = {}
# with open(playerDataFile, 'r') as f:
#     reader = csv.DictReader(f)
#     reader = sorted(reader, key=operator.itemgetter(9), reverse=False)
#     for row in reader:
#         if row['Season'] == '2016-17' :
#             firstSeasonPlayers.append(row)
#         else:
#             secondSeasonPlayers.append(row)

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

print("Collecting training data")
numTrainingSamples, trainingData, trainingValues = createMatrices(numDataPoints, numVariables, gameInfoDict, viewersPerGame)

# Runs the regression
ourModel = linear_model.LinearRegression()
ourModel.fit(trainingData, trainingValues)

# Compares the trained scores on the original set to their actual values using r^2
print("Score on training set: {}".format(ourModel.score(trainingData, trainingValues)))

resultsOnTrainingSet = ourModel.predict(trainingData)
print("MAPE on training set: {}".format(MAPE(resultsOnTrainingSet.flatten().tolist(), trainingValues.flatten().tolist())))


### TESTING ###

print("Collecting testing data")
numTestSamples, testData, testGameIdxDict = createMatrices(numTestDataPoints, numVariables, testGameInfoDict)

# Predict viewership
resultsOnTestSet = ourModel.predict(testData)

# Write prediction results to a file
with open(testFile, 'r') as testF:
    reader = csv.DictReader(testF)
    headers = reader.fieldnames
    with open(outFile, 'w') as outF:
        writer = csv.DictWriter(outF,fieldnames=headers)
        writer.writeheader()
        for line in reader:
            gameId = line['Game_ID']
            gameIdx = testGameIdxDict[gameId]
            line['Total_Viewers'] = int(resultsOnTestSet[gameIdx][0])
            writer.writerow(line)





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