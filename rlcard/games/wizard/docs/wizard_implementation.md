## The Goal

The goal is to create a fully functional AI of The Wizard Cardgame. 

There are two ways in which the approach of reinforcement learning for the cardgame will work. 
The first and easier idea is to mix rule based approaches with reinforcement learning. 
* The forcasting round within wizard will be rule based.
* The actual rounds will be RL based Models.
The second approach is that only reinforcement learning is used also for the forcasting for every round played.
* The forcasting round within wizard will be on RL based Models.
* The actual rounds will be on RL based Models.

## How to Forcast

An existing problem that it is not possible to know which card will get the trick for sure. So the forcast is very hard to learn for the reinforcement algorithm. The heuristic approach only works good the less cards you start with in Hand. So the more cards you have in the hand the worse it gets. Due to the game it will bbe possible up to 20 tricks.

the heuristic is not yet implemented!