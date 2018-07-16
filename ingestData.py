import csv
from collections import defaultdict
import datetime
import numpy as np
import operator
from sklearn import datasets, linear_model, neural_network
# from sklearn.neural_network import MLPClassifier

analyticsDataPath = "./BusinessAnalyticsData/"
scrapedDataPath = "./ScrapedData/"

gameDataFile = analyticsDataPath + "game_data.csv"
playerDataFile = analyticsDataPath + "player_data.csv"
trainingFile = analyticsDataPath + "training_set.csv"
testFile = analyticsDataPath + "test_set.csv"
outFile = "predicted_viewership.csv"
national201617File = scrapedDataPath + "2016-2017_nba_nat_tv.txt"
national201718File = scrapedDataPath + "2017-2018_nba_nat_tv.txt"


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
        data[data_point][winDifferentialIdx] = game['differential']
        data[data_point][espnNationalIdx] = game['espn']
        data[data_point][abcNationalIdx] = game['abc']
        data[data_point][tntNationalIdx] = game['tnt']
        data[data_point][nbatvNationalIdx] = game['nbatv']
        data[data_point][dateInSeasonIdx] = game['date-season']
        data[data_point][nbatvNationalIdx + game['weekday']] = 1

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

### MICHAEL DETERMINE IF NATIONAL TV GAME ###

monthsofFirstHalf = {'September' : 9, 'October' : 10, 'November' : 11, 'December' : 12 }
monthsofSecondHalf = {'January' : 1, 'February' : 2, 'March' : 3, 'April' : 4}
teamCodes = {'Knicks' : 'NYK',  'Cavaliers' : 'CLE', 'Spurs' : 'SAS', 'Warriors' : 'GSW', 'Thunder' : 'OKC', 'Sixers': 'PHI', 'Rockets' : 'HOU', 'Lakers' : 'LAL', 'Celtics' : 'BOS', 'Bulls' : 'CHI', 'Clippers' : 'LAC', 'Blazers': 'POR', 'Raptors' : 'TOR', 'Mavericks' : 'DAL',  'Nets' : 'BKN', 'Heat' : 'MIA', 'Kings' : 'SAC', 'Grizzlies' : 'MEM', "T'Wolves" : 'MIN', 'Jazz' : 'UTA', 'Pelicans' : 'NOR', 'Hornets' : 'CHA', 'Bucks' : 'MIL', 'Nuggets' : 'DEN', 'Hawks' : 'ATL', 'Wizards' : 'WAS', 'Pacers' : 'IND', 'Pistons' : 'DET', 'Magic' : 'ORL', 'Suns' : 'PHX', 'Orlando': 'ORL', 'Cavs' : 'CLE'}
christmas = '12/25/2016'
openingday = '10/25/2016'
christmas2017 = '12/25/2017'
openingday2017 = '10/17/2017'
networks = ['espn', 'abc', 'tnt', 'nbatv']

nat_tv_schedule = {}
with open(national201617File, 'r') as f:
    day = None
    month = None
    date = None
    full_date = None
    for line in f:
        line_arr = line.split()
        if line[0].isalpha():
            day = line_arr[0].strip(',')
            month = line_arr[1]
            date = line_arr[2]
            if month in monthsofFirstHalf:
                full_date = str(monthsofFirstHalf[month]) + '/'  + date + '/' + '2016'
            else:
                full_date = str(monthsofSecondHalf[month]) + '/' + date + '/' + '2017'
                isOpeningDay = False
        else:
            isESPN = 0
            isNBATV = 0
            isABC = 0
            isTNT = 0
            teams = line_arr[2].split('/')
            teams.sort()
            key0 = full_date + '_' + teamCodes[teams[0]]
            key1 = full_date + '_' + teamCodes[teams[1]]
            network = line_arr[-1]
            if network == 'ESPN':
                isESPN = 1
            elif network == 'ABC':
                isABC = 1
            elif network == 'TNT':
                isTNT = 1
            elif network == 'TV':
                isNBATV = 1
            retRow = [isESPN, isABC, isTNT, isNBATV]
            nat_tv_schedule[key0] = retRow
            nat_tv_schedule[key1] = retRow

with open(national201718File, 'r') as f:
    day = None
    month = None
    date = None
    full_date = None
    nextLine = False
    saveKey = None
    for line in f:
        line_arr = line.split()
        if line[0].isalpha() and line_arr[0] != 'Only':
            day = line_arr[0].strip(',')
            month = line_arr[1]
            date = line_arr[2]
            if month in monthsofFirstHalf:
                full_date = str(monthsofFirstHalf[month]) + '/' + date + '/' + '2017'
            else:
                full_date = str(monthsofSecondHalf[month]) + '/' + date + '/' + '2018'
        else:
            isESPN = 0
            isNBATV = 0
            isABC = 0
            isTNT = 0
            if not nextLine:
                teams = line_arr[2].split('-')
                teams.sort()
                key0 = full_date + '_' + teamCodes[teams[0]] 
                key1 = full_date + '_' + teamCodes[teams[1]]
                network = line_arr[-1]
                if network == 'ESPN':
                    isESPN = 1
                elif network == 'ABC':
                    isABC = 1
                elif network == 'TNT':
                    isTNT = 1
                elif network == 'TV':
                    isNBATV = 1
                else:
                    nextLine = True
                    saveKey0 = key0
                    saveKey1 = key1
                retRow = [isESPN, isABC, isTNT, isNBATV]
                nat_tv_schedule[key0] = retRow
                nat_tv_schedule[key1] = retRow
            else:
                nextLine = False
                network = line_arr[-1]
                if network == 'ESPN':
                    isESPN = 1
                elif network == 'ABC':
                    isABC = 1
                elif network == 'TNT':
                    isTNT = 1
                elif network == 'TV':
                    isNBATV = 1
                
                retRow = [isESPN, isABC, isTNT, isNBATV]
                nat_tv_schedule[saveKey0] = retRow
                nat_tv_schedule[saveKey1] = retRow

### END OF MICHAEL NATIONAL GAME ###


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
# {gameID: {'H':{homeTeamInfo}, 'A':{awayTeamInfo}, 'differential':int(gameWinDiff), 'abc':0/1, etc, 'weekday':0-6}}
gameInfoDict = {}
testGameInfoDict = {}
with open(gameDataFile, 'r') as f:
    reader = csv.DictReader(f)
    date16 = datetime.datetime(2016, 10, 25)
    date17 = datetime.datetime(2017, 10, 17)
    for row in reader:
        team = row['Team']
        teams.add(team)
        gameId = row['Game_ID']
        date = row['Game_Date']
        month, day, year = date.split('/')
        currDatetime = datetime.datetime(int(year), int(month), int(day))
        weekday = currDatetime.weekday()
        season = row['Season']
        if season == '2016-17':
            timeDelta = (currDatetime - date16).days
            date16 = currDatetime
        elif season == '2017-18':
            timeDelta = (currDatetime - date17).days
            date17 = currDatetime
        else:
            print("WARNING: season {} incorrectly formatted".format(season))
        key = date + '_' + team
        nat_tv_info = [0,0,0,0]
        # print(key)
        if key in nat_tv_schedule:
            nat_tv_info = nat_tv_schedule[key]

        # Only gathering training currently
        if gameId in testSetGameIds:
            if gameId in testGameInfoDict:
                currentData = testGameInfoDict[gameId]
                currentData['differential'] = abs(team1wins - int(row['Wins_Entering_Gm']))
            else:
                currentData = {}
                team1wins = int(row['Wins_Entering_Gm'])
                numTestDataPoints += 1
        else:
            if gameId in gameInfoDict:
                currentData = gameInfoDict[gameId]
                currentData['differential'] = abs(team1wins - int(row['Wins_Entering_Gm']))
            else:
                currentData = {}
                team1wins = int(row['Wins_Entering_Gm'])
                numDataPoints += 1
        
        # loc can be 'H' or 'A'
        loc = row['Location']
        currentData[loc] = {}

        winsEntering = int(row['Wins_Entering_Gm'])
        lossesEntering = int(row['Losses_Entering_Gm'])
        currentData[loc]['wins'] = winsEntering
        currentData[loc]['losses'] = lossesEntering
        currentData[loc]['team'] = row['Team']
        currentData['weekday'] = weekday
        currentData['date-season'] = timeDelta

        i = 0
        for network in networks:
            currentData[network] = nat_tv_info[i]
            i += 1

        if gameId in testSetGameIds:
            testGameInfoDict[gameId] = currentData
        else:
            gameInfoDict[gameId] = currentData

### Gather the top five percent of players for each season 

# splits player_data into '17 and '18 season
firstSeasonPlayerData = []
secondSeasonPlayerData = []
with open(playerDataFile, 'r') as f:
    reader = csv.DictReader(f)
    reader = sorted(reader, key=operator.itemgetter('Points'), reverse=False)
    for row in reader:
        if row['Season'] == '2016-17' :
            firstSeasonPlayerData.append(row)
        else:
            secondSeasonPlayerData.append(row)


### ABOVE ARE DEFINITIONS, READING IN INITIAL DATA ###
### BELOW IS SETTING UP, TRAINING, AND TESTING THE MODEL ###

teamsSorted = sorted(list(teams))
numTeams = len(teams)
# Column plan: sorted teams for binary 1/0, followed by home team win, home team loss, away team win, away team loss, 
# win differential, national tv airings, weekday (0-6)
# weekday treated categorically
numVariables = numTeams + 10 + 7
homeWinIdx = numTeams
awayWinIdx = numTeams + 1
homeLossIdx = numTeams + 2
awayLossIdx = numTeams + 3
winDifferentialIdx = numTeams + 4
espnNationalIdx = numTeams + 5
abcNationalIdx = numTeams + 6
tntNationalIdx = numTeams + 7
nbatvNationalIdx = numTeams + 8
dateInSeasonIdx = numTeams + 9




### TRAINING ###

print("Collecting training data\n")
numTrainingSamples, trainingData, trainingValues = createMatrices(numDataPoints, numVariables, gameInfoDict, viewersPerGame)

# Runs the regression
print("Linear Regression\n")
ourModelLinear = linear_model.LinearRegression()
ourModelLinear.fit(trainingData, trainingValues)

# Compares the trained scores on the original set to their actual values using r^2
print("Score on training set: {}".format(ourModelLinear.score(trainingData, trainingValues)))

resultsOnTrainingSet = ourModelLinear.predict(trainingData)
print("MAPE on training set: {}".format(MAPE(resultsOnTrainingSet.flatten().tolist(), trainingValues.flatten().tolist())))

### NEURAL NETWORK NOT WORKING, WONT IMPORT ###
# print("\nNeural Network\n")
# ourModelNN = neural_network.MLPRegressor(hidden_layer_sizes=(numVariables, numVariables), max_iter=500)
# ourModelNN.fit(trainingData, trainingValues)
# resultsOnTrainingSet = ourModelNN.predict(trainingData)
# print("MAPE on training set: {}".format(MAPE(resultsOnTrainingSet.flatten().tolist(), trainingValues.flatten().tolist())))

### TESTING ###

print("Collecting testing data")
numTestSamples, testData, testGameIdxDict = createMatrices(numTestDataPoints, numVariables, testGameInfoDict)

# Predict viewership
resultsOnTestSet = ourModelLinear.predict(testData)

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