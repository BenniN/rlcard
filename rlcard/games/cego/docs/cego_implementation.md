## The Goal

The goal is to create a fully functional AI of cego. This AI will be implemented within **Cego Online Reloaded**. 

The idea is to mix rule based approaches with reinforcement learning.
* The betting round within cego will be rule based.
* Also the system behind keeping and throwing away cards for the specific cego sub variants will be rule based.
* The actual sub games will be RL based Models.

## When to play cego

An existing problem is that the cego player also plays cegos, that an expert player usually wouldn't play. This puts the cego player in a disadvantageous position. This disadvantage might also negatively affect the AI training. So the idea is to implement a simple heuristic that negates that disadvantage.

From [cegofreunde_st_georgen_taktik_2012](http://cegofreunde.jimdofree.com/taktik-und-tipps/) we get that cego should be played when the player has 15 - 17 points on his hand. 

The lower bound (15) was taken as a baseline for the training of the AI.