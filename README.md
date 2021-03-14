# Recommender Systems Competition @ Politecnico di Milano

<p align="center">
    <img src="https://i.imgur.com/mPb3Qbd.gif" width="180" alt="Politecnico di Milano"/>
</p>

The Recommender System Course at Politecnico di Milano is divided into two parts: the most important one consists in a Internal Kaggle Competition amongs students.

This repository is forked by the [official one](https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi) of Polimi, developed by [Maurizio Ferrari Dacrema](https://mauriziofd.github.io/), Postdoc researcher at Politecnico di Milano. See the [website](http://recsys.deib.polimi.it/) for more information on its research group.


## The Competition
The application domain is book recommendation. The datasets contains both interactions between users and books, and tokens extracted from the book text. The main goal of the competition is to discover which item a user will interact with.



## What you can find in this repo
 Only what is necessary to build and test the model we used to reach our best result in the competition (**13th place**).
 In particular the recommender systems algorithm that you can find are:
 * Item based CF
 * Graph Based RP3Beta
 * ALS (the ALS Class wrap the als class of python [implict library](https://implicit.readthedocs.io/en/latest/als.html))
 
 For the implementation of other recommender system algorithms look at the original repository in the [References](#References)
 
## The Best Model:
 Desciption of our final (best) model.
 
 <p align="center">
    <img src="https://i.imgur.com/p4SBP8b.png" width="500" alt="model schema"/>
</p>

Divided in 4 layers. Each layer represent a set of users based on the number of interactions they have:
 - \[0,5) represents users with number of interaction lower than 5, and the model for these user consists in an **Item-Based CF**
 - \[5, 10): number of interactions from 5 to 10, the model is an **hybrid between an Item-Based CF and an ALS**
 - \[10, 50): number of interactions from 10 to 50, the model is an **hybrid between an Item-Based CF (the same of the first layer) and an ALS**
 - \[50, ...): number of interactions higher or equal to 50, the model is an **hybrid between an Item-Based CF and an RP3Beta**
 
## Parameter tuning:
A simple wrapper of scikit-optimize allowing for a simple and fast parameter tuning.
In particular this wrapper give the possibility to do more tests (choose the value of n_tests) on different data split, in order to find the best parameters in a more reliably way. Avoid the overfitting obtained by optimizing on a specific test split.
 
## This repository contains the following runnable scripts

 - evaluation_and_submission.py.py: Script running the best model described before. You can choose to run only the evaluation (return some metrics of the model, like MAP) or only the submission procedure (return the csv file we had to submit to kaggle) or both of them.
 - parameter_tuning.py: Script performing parameter tuning for the chosen algorithm. By default the algorithm is ALS, but by modifying the code you can evaluate also the others algorithm.
 

## Requirements

Install all the requirements and dependencies using the following command.
```console
pip install -r requirements.txt
```

## Results
The competition has two deadlines: the first one after one month, the second (and final) one after two months.

We obtained the following results in the two deadline:
* First Deadline:
    * *public leaderboard*: **8th** position, score: 0.0952615
    * *private leaderboard*: **8th** position, score: 0.102143
* Final Deadline:
    * *public leaderboard*: **13th** position, score: 0.0963792
    * *private leaderboard*: **13th** position, score: 0.104061

## Team
* Manuel Salamino ([github](https://github.com/manuelsalamino))
* Tommaso Fioravanti ([github](https://github.com/tommasofioravanti))
