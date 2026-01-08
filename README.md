# Riichi Mahjong Expected Value Calculator

## Introduction

This tool aims to calculate expected values at the start of a round given a game state - (Wind, Round, Honba, Riichi sticks, Player Scores)

Uses [Houou](https://github.com/NikkeTryHard/tenhou-to-mjai/releases/tag/v1.2.0) games from Tenhou, with it's uma system (90/45/0/-135).

We define a final score as the ```raw score - 25 + uma```. The expected value is the average final score a player can expect to achieve from a given game state.

For example, starting at S4 with ```0/20/32.5/45``` should return something similar to 
```-135/-5/52.5/110```, because of the difficulty of changing placements in the last round.
Therefore ```f(1,4,0,0,0,15000,30000,55000) ~= (-135,-5,52.5,110)```

However, the same scores at S1 should return a tighter distribution, as placements can change more easily. ```f(1,1,0,0,0,15000,30000,55000) ~= (-90,-25,12.5,90)```. (Random numbers for illustration only). This agrees with intuition - being in last place at the start of the game is not as bad as being in last place at the end of the game.

## What is this useful for?

I don't expect this tool to be used during actual gameplay, but instead to help understand the value of point differences in various situations.

Say you are at the start of South 3, leading.

```
S3
26000
32000 (You)
26000 (Dealer)
16000
```

- What is your probability of finishing first? Or more generally, what is your EV?
- How much does getting an extra 2000 points help you solidfy first place? How much can you risk dealing in?

Or let's say you're in last place, in South 2.

```
S2
26000
32000 (Dealer)
26000 
16000 (You)
```
- Should you confirm a 2600 point hand, or risk a 5200 point hand to close the gap further?

Knowing the expected value of the game depending on the round result allows you to make these decisions more optimally.

**A common misconception when talking with other players is that they argue your hand completely changes the EV** - this tool is not used to calculate the EV of a current game state, but rather the EV at the start of the next round. I think an example shows this best.

## Example

In all last, calculating the effects of point changes is very easy.

For example, lets say you are South in South 4, with the following scores:

S4
15000
20000 (You)
25000
40000 (Dealer)

Let's create a simplified scenario. Assume you and dealer are pushing - everyone else has folded.
You have two options:

Push:
- 50% chance of winning 8k. 
- 50% chance of losing 8k. 

Fold:
- nothing happens (the dealer tsumo score loss is negligible)

What should you choose?

Given tenhou uma, the answer is obvious - fold.
On losing 8k you drop to last place, which is a ```-(135+8) = -142``` point swing.
On winning 8k you only gain ```(45+8) = 53``` points.

You can also calculate the threshold for pushing:
```
let p = probability of winning.
Expected Value of pushing = p * 53 + (1-p) * -142
Setting this equal to 0 and solving for p gives
p = 142 / (142 + 53) = 72.8%
```


However, what if you are in this exact position, but in South 3?

S3
15000
20000 (You)
40000 (Dealer)
25000

If you win, (let's assume you ron Dealer), at the start of round 4, the scores will be:

S4
15000
28000 (You)
32000 
25000 (New Dealer)

Lose,

S4
15000
12000 (You)
48000
25000 (New Dealer)

The expected value of these scores are now more murky. When winning, you now have a chance to get 1st place, as well a chance to drop to third. In the second case, you are not necessarily last place - you could still overtake Player 3.

The positives and negatives of pushing are now less clear. This is where this tool comes in - by estimating the EV of each future scenario, you can make a more informed decision in the present.

Of course, this is a simplified example. In reality, you have to consider the chances of other players winning the current hand, and the chances of tsumo vs ron, etc. I don't ever expect to use this tool for actual gameplay, but instead to understand general trends and the value of point differences in various situations.

## Implementation

Lowkey, this is my first time using machine learning or doing any data science stuff like this. Asking ChatGPT to build the model was the easy part - I'm struggling with how to properly evaluate the model. I'm like PRETTY SURE its RMSE but I swear I'm being gaslighted by ChatGPT saying it's MAE.

More detailed modeling/evaluation notes: `docs/ev_modeling_notes.md`.




