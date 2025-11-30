This repo aims to calculate expected values at the start of a round given the game state -
- Wind, Round, Honba, Riichi sticks, Player Scores

Using tenhou uma - 90/45/0/-135.


Say you are at the start of South 3, leading.

S3
26000
32000 (You)
26000 (Dealer)
16000

What is your probability of finishing first? Or more generally, what is your EV?

How much does getting an extra 2000 points help you solidfy first place? How much can you risk dealing in?

Or let's say you're in last place, in South 2.

S2
26000
32000 (Dealer)
26000 
16000 (You)

Should you confirm a 2000 point hand, or risk a 5200 point hand to close the gap further?

Knowing the expected value of the game depending on the round result allows you to make these decisions more optimally.

Examples:

South 4 with 0 honba and 0 riichis, and scores: 0,15000,30000,55000

Current values (assuming no change in score):
-25 + (-135) = -160
-10 + 0 = -10
5 + 45 = 50
30 + 90 = 120

-160, -10, 50, 120

The expected value will be very close to the current standings because it would be very difficult to change the placements in the last round.

However, if the scores were closer together, for example: 15000, 20000, 30000, 35000, the expected value would be different. Player 2 would have a lower EV because of the risk of dropping to last place, and this EV is given to Player 1.

I aim to calculate these expected values for all rounds and all game states.


Usage:

In all last, calculating the effects of point changes is very easy.
For example, lets say you are South in South 4, with the following scores:

S4
15000
20000 (You)
25000
40000 (Dealer)

To simplify, let's assume you and dealer are pushing - everyone else has folded.
You have two options:

Push:
- 50% chance of winning 8k.
- 50% chance of losing 8k.

Fold:
- nothing happens (the dealer tsumo score loss is negligible)

What should you choose?

Given tenhou uma, the answer is obvious - fold.
On losing 8k you drop to last place, which is a -135 point swing.
On winning 8k you only gain a +45 point swing.

You can also calculate the threshold for pushing - when you have a 75% chance of winning, as 0.75 * 45 + 0.25 * -135 = 0.


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

The expected value of these scores are now more murky. When winning, you now have an increased chance to get 1st place. In the second case, you are not necessarily last place - you could still overtake Player 3.

Since these both benefit you, the threshold for pushing is lower than 75%.

Knowing the expected value of both situations will allow you to calculate the exact threshold for pushing or folding in this situation.



Of course, this is a simplified example. In reality, you have to consider the chances of other players winning, and the chances of tsumo vs ron, etc. I don't ever expect to use this tool for actual gameplay, but instead to understand general trends and the value of point differences in various situations.

