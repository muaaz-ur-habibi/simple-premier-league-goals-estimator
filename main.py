import torch.nn as nn
import torch.optim as optimizer

# for playing with the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

import random as rd

import math


# the data model class
# loads and does operations on the data file
class DataModel:
    def __init__(self,
                 data_file:str):
        
        self.data_file=data_file

    def read_data(self):
        file = pd.read_csv(self.data_file)
        dataframe = pd.DataFrame(file)
        
        return dataframe
    
    # for this model i wanna get only liverpool's data
    def data_extractor(self):
        # cleaned data contains a dictionary of data of all the rows that have liverpool
        cleaned_data = []

        data = self.read_data()

        for i in range(data.shape[0]):
            cleaned_data.append(
                    {
                        'home_team': str(data.loc[i, "Home"]).lower(),
                        'away_team': str(data.loc[i, "Away"]).lower(),
                        'home_team_coded': self.teamnames_to_digits(str(data.loc[i, "Home"]).lower()),
                        'away_team_coded': self.teamnames_to_digits(str(data.loc[i, "Away"]).lower()),
                        'home_goals': data.loc[i, "HomeGoals"],
                        'away_goals': data.loc[i, "AwayGoals"],
                        'year': data.loc[i, "Season_End_Year"],
                        'probabilities': self.scores_to_probabilities(data.loc[i, "HomeGoals"],
                                                                              data.loc[i, "AwayGoals"])
                    }
                )

        return cleaned_data
    
    def visualize(self):
        data = self.data_extractor()

        #home = [i['home_team'] for i in data]
        sample_home = 'manchester city'
        sample_away = 'leicester city'
        sample_data = []
        
        for i in data:
            if i['home_team'] == sample_home and i['away_team'] == sample_away:
                sample_data.append((i['home_goals'], i['away_goals'], i['away_team'], i['year']))

        x = [i[3] for i in sample_data]
        y = [i[0] for i in sample_data]

        plot.plot(x, y, color='red', marker='s')
        plot.show()

        return sample_data


    def teamnames_to_digits(self, teamname):
        return [ord(char)-7 for char in teamname]

    
    def scores_to_probabilities(self,
                                home_score:int,
                                away_score:int):
        
        total = home_score + away_score

        home_probab = home_score / total
        away_probab = away_score / total

        # we are assumming that  score of 0-0 means a 50 50 chance of either team winning
        if not math.isnan(home_probab):
            return home_probab, away_probab
        else:
            return 0.5, 0.5


class EstimatorModel(nn.Module):
    def __init__(self,
                 input_neurons:int,
                 hidden_neurons:int,
                 output_neurons:int):

        super().__init__()
        self.input_neurons= nn.Linear(in_features=input_neurons, out_features=hidden_neurons)
        self.hidden_neurons= nn.Linear(in_features=hidden_neurons, out_features=hidden_neurons)
        self.output_neurons= nn.Linear(in_features=hidden_neurons, out_features=output_neurons)

        # activation functions
        # trying to use all 3 of these
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        #self.softmax = nn.functional.softmax()


    def forward_sigmoid(self, x):
        x = self.input_neurons(x)
        x = self.hidden_neurons(x)
        x = self.output_neurons(x)
        
        x = self.sigmoid(x)
        return x


cleaned_data = DataModel('data/premier-league-matches.csv').data_extractor()


# setting model parameters
inputs = 2
hiddens = 10
outputs = 2
epochs = 11
batch_size = 5


# creating the model, criterion and optimimzers
model = EstimatorModel(input_neurons=inputs,
                       hidden_neurons=hiddens,
                       output_neurons=outputs)

criterion = nn.MSELoss()
optim = optimizer.Adam(params=model.parameters(), lr=0.002)

for e in range(epochs):
    # iterate over the batch size
    for b in range(0, len(cleaned_data), batch_size):
        home_teams = [i['home_team_coded'] for i in cleaned_data]
        home_teams = home_teams[b: batch_size+1]
        away_teams = [i['away_team_coded'] for i in cleaned_data]
        away_teams = away_teams[b: batch_size+1]

        home_goal = [i['home_goals'] for i in cleaned_data]
        home_goal = home_goal[b: batch_size+1]
        away_goal = [i['away_goals'] for i in cleaned_data]
        away_goal = away_goal[b: batch_size+1]

        teams = []
        scores = []

        for i in range(len(home_teams)):
            teams.append([home_teams[i], away_teams[i]])

            scores.append([home_goal[i], away_goal[i]])
        
        #prediction = model()

        print(teams)

        quit()