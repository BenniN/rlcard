# Wizard RLCard

# Wizard Environment

## The Game Classes

Folder: *rlcard/games/wizard/*

* **WizardCard**: the card class
* **WizardDealer**: the class for dealing cards
* **WizardPlayer**: the player class
* **WizardRound**: represents a round of Wizard
* **WizardJudger**: Judges the round of Wizard
* **WizardGame**: represents a game of Wizard (abstract)
* **WizardStandardGame**: Implementation of WizardGame for the standard version of *Wizard*
* **WizardCompleteInformationGame**: Implementation of WizardGame for a complete Information Game of *Wizard*
* **Utils**: helper functions

### Card Encoding

File: ./jsondata/action_space.json

"\[rank\]-\[suit\]"

| Index |     Cards     |
|:-----:|:-------------:|
| 0-12  |  Red Colour   |
| 13-25 | Green Colour  |
| 26-38 |  Blue Colour  |
| 39-51 | Yellow Colour |
| 52-55 |    Narren     |
| 56-59 |    Wizard     |

## State Representation

|            Key             |                          Description                           |                  Example                   |
|:--------------------------:|:--------------------------------------------------------------:|:------------------------------------------:|
|          **hand**          |     The cards, the player currently has on his hand: List      |       ['1-r', '13-y', 'n-1', 'w-4']        |
|         **trick**          |             The cards currently in the trick: List             |       ['1-r', '13-y', 'n-1', 'w-4']        |
| **predicted_trick_cards**  |             The cards who should get a trick: List             |       ['1-r', '13-y', 'n-1', 'w-4']        |
| **predicted_trick_amount** | The amount of tricks the player wants to score this round: int |                     3                      |
|         **target**         |              The card that has to be served: str               |                   '5-g'                    |
|      **winner_card**       |          The card that currently wins the trick: str           |                   'w-4'                    |
|     **winner_player**      |         The player that currently wins the trick: int          |                     3                      |
|     **legal_actions**      |         The action the player is allowed to play: List         |               ['1-g', 'n-1']               |
|      **start_player**      |         The player who started the current round: int          |                     0                      |
|      **num_players**       |               The Number of players playing: int               |                     3                      |
|     **current_player**     |                    The current player: int                     |                     2                      |
|  **current_trick_winner**  |          The player who currently wins the trick: int          |                     1                      |
|     **played_tricks**      |       The tricks that have already been played: 2D-List        | ['1-r', '13-y', 'n-1', 'w-4'], [...], ...] |
|   **last_round_winner**    |               The winner of the last round: int                |                     3                      |

Detailed encoding needs to be implemented

| indexes |                                     Description                                     |
|:-------:|:-----------------------------------------------------------------------------------:|
|    -    |                            The cards on the players hand                            |
|    -    |                The cards who where responsible for predicted tricks                 |
|    -    |                       The card that currently wins the trick                        |
|    -    |                       All the cards within the current trick                        |
|    -    | All the cards that haven't been played jet and may still be played by other players |


## Classes

### Card

This class contains the static attribute info, which contains information about the card suits, ranks and values of the cards, as well as a couple static helper methods for printing and comparing.

#### Attributes

* **suit**: the suit of the card
* **rank**: the rank of the card

### Round 

This class represents a single round. The first player of the round is the winner of the previous round. in the very first round it is random who will start. A Game consists between 10 to 20 rounds depending on the amount of players. Allowed are Player amounts between 3 and 6 People

### Judger

The judger class generates the payoffs for the players and rl algorithms.

Only if the player got the same amount of tricks as he predicted he will be rewarded with points. else the player will lose points.

### Dealer

The object, responsible for shuffling and dealing the cards to the players starting with one card each n the first round. In the following rounds it will be alway one card more than the turn before

### Player

This class represents a player in the game. All players are the same

### Utils

Helper functions are within file **utils.py**.

### Game

This class represent the game itself and manages the full game state. This class is abstract. Usually you will play the standard game of wizard where you do not know the cards of your "rival" players.
Currently following Variants are planned to be fully implemented:

* **WizardStandardGame**: The Standard Game of wizard
* **WizardCompleteInformationGame**: The Variant where all Cards are known to each player, it will be as a reference of how accurate the rl will be as imperfect information to perfect information games
* TODO: Implement both variants

### Testing

**test.py** contains the test cases for the game class.
* TODO: Not done yet

## The Environment Class

Path: *rlcard/envs/wizard.py*

### Configuration

The Environment will allow the following configurations:
* **game_variant** (str): The setting will allow 2 values, "standard" and "full".
* **game_judge_by_points** (int): 
  * *0* (default): The payoffs of the game will be judged by the points the players gets when he predicted the tricks correctly. 
  * *1*: The payoffs will be judged by weather the player wins or loses. The Player will be rewarded with the points for every trick he received independent of what he predicted.

