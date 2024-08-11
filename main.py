import torch.nn as nn
import torch.optim as optimizer
from torch import tensor, eye, stack, no_grad, save

# for playing with the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot


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
                        'home_goals': data.loc[i, "HomeGoals"],
                        'away_goals': data.loc[i, "AwayGoals"],
                        'year': data.loc[i, "Season_End_Year"],
                        'probabilities': self.scores_to_probabilities(data.loc[i, "HomeGoals"],
                                                                      data.loc[i, "AwayGoals"])
                    }
                )
        # training till 2022 year so that we can
        # evaluate it for the 2023 year
        # NOTE: comment this line out when you have evaluated it and are satisfied with the results
        cleaned_data = cleaned_data[:11646]

        # one hot encoding labels
        names = [i['home_team'] for i in cleaned_data]
        onehot_dict = self.one_hot_encoding_labels(names)

        for i in cleaned_data:
            for teamname, onehot in onehot_dict.items():
                if teamname == i['home_team']:
                    i['home_team_onehot'] = onehot
                
                if teamname == i['away_team']:
                    i['away_team_onehot'] = onehot
        

        return cleaned_data
    
    def visualize(self):
        data = self.data_extractor()

        sample_home = 'manchester city'
        sample_away = 'leicester city'
        sample_data = []
        
        for i in data:
            if i['home_team'] == sample_home and i['away_team'] == sample_away:
                sample_data.append((i['home_goals'], i['away_goals'], i['away_team'], i['year']))

        x = [i[3] for i in sample_data]
        y = [i[0] for i in sample_data]

        plot.title(f"{sample_home}'s goals against {sample_away}")
        plot.plot(x, y, color='red', marker='s')
        plot.show()

        return sample_data
    
    def one_hot_encoding_labels(self, labels:list):

        size = len(labels)
        vectors = eye(size)
        onehot_dict = {word: vectors[i] for i, word in enumerate(labels)}

        return onehot_dict

    
    def scores_to_probabilities(self,
                                home_score:int,
                                away_score:int):
        np.seterr(divide='ignore', invalid='ignore')
        
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
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.input_neurons(x)
        x = self.relu(x)

        x = self.hidden_neurons(x)
        x = self.relu(x)

        x = self.hidden_neurons(x)
        x = self.relu(x)

        x = self.output_neurons(x)

        x = self.relu(x)
        return x

print("[INFO] Creating data for training")
data = DataModel('data/premier-league-matches.csv')
cleaned_data = data.data_extractor()

input_data = []
output_data =[]

print("[INFO] Further cleaning and processing data")
for i in range(len(cleaned_data)):
    input_data.append((cleaned_data[i]['home_team_onehot'], cleaned_data[i]['away_team_onehot']))
    output_data.append((tensor([cleaned_data[i]['home_goals']]), tensor([cleaned_data[i]['away_goals']])))


print("[INFO] Setting model parameters")
# setting model parameters
# NOTE: input_size should be same as however many rows
# of the csv file you are training the model on
input_size = 11646
hidden_size = 25
output_size = 1
epochs = 20
batch_size = 5


print("[INFO] Creating the model, loss function and optimizer")
# creating the model, criterion and optimimzers
model = EstimatorModel(input_neurons=input_size,
                       hidden_neurons=hidden_size,
                       output_neurons=output_size)

loss_func = nn.MSELoss()
optim = optimizer.Adam(params=model.parameters(), lr=0.007)

print("[INFO] Starting training loop")

try:
    for e in range(epochs):
        x=0
        for inputs, outputs in zip(input_data, output_data):
            inputs = stack(inputs)
            outputs = stack(outputs)

            prediction = model(inputs)

            loss = loss_func(prediction, outputs.float())

            optim.zero_grad()
            loss.backward()
            optim.step()

            x+=1
            # comment this print statement if you dislike too much verbosity
            #print(f"[INFO] {x+1} pair completed. Loss: {loss}")

        print(f"[TRAINING] Epochs finished: {e+1}/{epochs}. Loss: {loss}")
except KeyboardInterrupt:
    pass

print("[INFO] Saving the model")
# save the trained model
save(obj=model, f='models/predictor.pt')

print("[INFO] Evaluating the model")
# the evaluation part
model.eval()

while True:
    team1, team2 = str(input("Enter two team names, seperated by a space: ")).split(" ")

    for dic in cleaned_data:
        if dic['home_team'] == team1:
            team1 = dic['home_team_onehot']
        
        elif dic['away_team'] == team2:
            team2 = dic['away_team_onehot']

    inp = stack((team1, team2))

    with no_grad():
        prediction = model(inp)

    print(prediction)