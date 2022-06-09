# Cego RLCard

# Cego Environment

## The Game Classes

Folder: *rlcard/games/cego/*

* **CegoCard**: the card class
* **CegoDealer**: the class for dealing cards
* **CegoPlayer**: the player class
* **CegoRound**: represents a round of Cego
* **CegoJudger**: Judges the round of Cego
* **CegoGame**: represents a game of Cego (abstract)
* **CegoGameStandard**: Implementation of CegoGame for the game variant of *Cego*
* **CegoSolo**: Implementation of CegoGame for the game variant of *Solo*
* **Utils**: helper functions

### Card Encoding

For know the part of the game, where the cego player selects the blind cards will be ignored for the reason of simplifying the action space.

File: ./jsondata/action_space.json

"\[rank\]-\[suit\]"

| Index | Cards |
|:-----:|:-----:|
| 0-7 | Clubs |
| 8-15 | Spades |
| 16-23 | Hearts |
| 24-31 | Diamonds |
| 32-53 | Trumps |

## State Representation

The state the player can possibly observe is the following (not all information is encoded within the observation jet).

| Key | Description | Example |
|:---:|:-----------:|:-------:|
| **hand** | The cards, the player currently has on his hand: List | ['7-c', 'k-s', '1-trump', 'gstieß-trump'] |
| **trick** | The cards currently in the trick: List | ['7-c', 'k-s', '1-trump', 'gstiess-trump']  |
| **target** | The card that has to be served: str | '11-trump' |
| **winner_card** | The card that currently wins the trick: str | 'gstiess-trump' |
| **winner_player** | The player that currently wins the trick: int | 2 |
| **valued_cards** | cards that the cego player (only the cego player knows of these) has layed aside: List | ['7-c', 'k-s', '1-trump', 'gstiess-trump' ...] |
| **legal_actions** | The action the player is allowed to play: List | ['12-trump', '14-trump'] |
| **start_player** | The player who started the current round: int | 0 |
| **num_players** | The Number of players playing: int | 4 |
| **current_player** | The current player: int | 2 |
| **current_trick_winner** | The player who currently wins the trick: int| 1 |
| **played_tricks** | The tricks that have already been played: 2D-List | [['7-c', 'k-s', '1-trump', 'gstieß-trump'], [...], ...] |
| **last_round_winner** | The winner of the last round: int | 3 |

The are several observation encodings possiblities implemented. Here is one example of a possible observation state encoding. The Size of this encoding is **224** or also **54x4+12**. The first **54x4** indexes encode card information. There are **54** cards within the game. The last **12** indexes encode the information about the current state of the game. 

The following table describes the encoding further:

| indexes | Description |
|:---:|:---:|
| 0-53 | The cards on the players hand |
| 54-107 | The card that currently wins the trick |
| 108-161 | All the cards within the current trick |
| 162-215 | All the cards that haven't been played jet and may still be played by other players |
| 216-227 | **This plane encodes other game specific information.** <br> **[216-219]**: The players within the same team have the value 1. <br> **[220-223]**: The player who would win the round is encoded. <br> **[224-227]**: The player who started the current trick round is encoded. |

There are alternative possible encodings for the observation state.

Theses are implemented as functions in the utils.py file. The Functions observation encoding functions are called **encode_observation_var[0-4]**. Additional information about the specific encoding can be found within the function documentation.

## Classes

### Card

This class contains the static attribute info, which contains information about the card suits, ranks and values of the cards, as well as a couple static helper methods for printing and comparing.

#### Attributes

* **suit**: the suit of the card
* **rank**: the rank of the card

### Round 

This class represents a single round. The first player of the round is the winner of the previous round. Cego consists of 11 rounds.

### Judger

The judger class generates the payoffs for the players.

If the player who receives the reward is the cego player only this player receives it. If the player who receives the reward is not the cego player, everybody but the cego player receives the reward.

### Dealer

The object, responsible for shuffling and dealing the cards to the players (11 each). This class also deals the blind cards (10 cards).

### Player

This class represents a player in the game. There is one special case of player, the cego player, which is indicated through the boolean-attibute **is_cego_player**. In this case the player gets a list **valued_cards**, which contains the cards the cego player has layed aside.

### Utils

Helper functions are within file **utils.py**.

### Game

This class represent the game itself and manages the full game state. This class is abstract. One Game is a single cego variant.
Currently following Variants are implemented:

* **CegoGameStandard**: The Variant Cego
  * A model of this variant can also be used to play all the sub games of cego.
* **CegoSolo**: The Variant Solo
* TODO: Implement the special games.

### Testing

**test.py** contains the test cases for the game class.

## The Environment Class

Path: *rlcard/envs/cego.py*

### Configuration

The Environment currently allows the following configurations:
* **game_variant** (str): The currently implemented variants.
  * "standard" (Cego) with heuristic
  * "solo"
  * "bettel"
  * "piccolo"
  * "utlimo"
  * ```diff ! TODO: Räuber Variant```
* **game_judge_by_points** (int): 
  * *0* (default): The payoffs of the game will be judged by the points the players gets. 
  * *1*: The payoffs will be judged by weather the player wins or loses. If the player loses the payoff is **0**, if the player wins the payoff is **1**.
  * *2*: The payoffs will be judged by weather the player wins or loses. If the player loses the payoff is **-1**, if the player wins the payoff is **1**.
* **game_activate_heuristic** (bool): take into account heuristic for the game variant if possible

