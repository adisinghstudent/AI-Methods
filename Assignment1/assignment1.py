
import math
from math import factorial

# Oppgave 1
print("\n=======================")
print(" Exercise 1: Part (a) ")
print("=======================\n")

def atomic_events(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))

# Output for Exercise 1a
print("Number of atomic events:")
print(atomic_events(52, 5))

print("\n=======================")
print(" End of Exercise 1: Part (a) ")
print("=======================\n")



# Part (b)
print("\n=======================")
print(" Exercise 1: Part (b) ")
print("=======================\n")

def atomic_events_odds(n, k):
    return 1 / (factorial(n) / (factorial(k) * factorial(n - k)))

# Output for Exercise 1b
print("Odds of a single atomic event:")
print(f"{atomic_events_odds(52, 5):.15f} %")  # no e-7

print("\n=======================")
print(" End of Exercise 1: Part (b) ")
print("=======================\n")



# Part (c)
print("\n=======================")
print(" Exercise 1: Part (c) ")
print("=======================\n")

def atomic_events_odds(n, k):
    return 1 / (factorial(n) / (factorial(k) * factorial(n - k)))

def flush_odds():
    return 4 / (atomic_events(52, 5)) # $ Flushes in a deck of 52, needs 5 cards

def four_of_a_kind_odds():
    return 13 / atomic_events(52, 4) # $ 13 different ranks in a deck of 52
    # i think this is wrong. 
# Output for Exercise 1c
print("Odds of a Royal Flush:")
print(f"{flush_odds():.15f} %\n")  # no e-7

print("Odds of Four of a Kind:")
print(f"{four_of_a_kind_odds():.15f} %")  # no e-7

print("\n=======================")
print(" End of Exercise 1: Part (c) ")
print("=======================\n")


#Exercise 2

Total_combinations = 4**(3)
print(Total_combinations)
bbb = 20 # 6
bebebe = 15 # 6
lll = 5 # 6
ccc = 3 # 6
cc = 2 # 8 
c = 1 # 16
none = Total_combinations - (6+6+6+6+8+16) # 16
print(none)

average = (bbb*6 + bebebe*6 + lll*6 + ccc*6 + cc*8 + c*16 + none*0) / Total_combinations
print(average)
oddswin = 48/64
print(oddswin)



import random
import statistics

def simulate_game(starting_coins, payout_average, win_probability, trials):
    total_plays = []
    for _ in range(trials):
        coins = starting_coins
        plays = 0
        while coins > 0:
            plays += 1
            coins -= 1  # Deduct 1 coin for playing
            if random.random() < win_probability:  # Check for a win
                coins += payout_average  # Add average payout
        total_plays.append(plays)
    return total_plays

# Parameters
starting_coins = 10
payout_average = 4.53  # Average payout per win
win_probability = 0.75  # Probability of winning
trials = 100

# Run simulation
results = simulate_game(starting_coins, payout_average, win_probability, trials)

# Compute mean and median
mean_plays = sum(results) / len(results)
median_plays = sorted(results)[len(results) // 2]

print(f"Mean number of plays: {mean_plays:.2f}")
print(f"Median number of plays: {median_plays:.2f}")