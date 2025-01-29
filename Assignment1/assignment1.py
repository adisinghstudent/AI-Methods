import math
from math import factorial
import random

# exercise 1: part (a)
def atomic_events(n, k):
    return factorial(n) / (factorial(k) * factorial(n - k))

print("atomic events:")
print(atomic_events(52, 5))

# exercise 1: part (b)
def atomic_events_odds(n, k):
    return 1 / (factorial(n) / (factorial(k) * factorial(n - k)))

print("odds of a single atomic event:")
print(f"{atomic_events_odds(52, 5):.15f}")

# exercise 1: part (c)
def flush_odds():
    return 4 / atomic_events(52, 5)

def four_of_a_kind_odds():
    return 13 / atomic_events(52, 4)

print("odds of a royal flush:")
print(f"{flush_odds():.15f}")
print("odds of four of a kind:")
print(f"{four_of_a_kind_odds():.15f}")

# exercise 2: part (a)
total_combinations = 4**3
bbb, bebebe, lll, ccc, cc, c = 20, 15, 5, 3, 2, 1
none = total_combinations - (6*bbb + 6*bebebe + 6*lll + 6*ccc + 8*cc + 16*c)
average = (bbb*6 + bebebe*6 + lll*6 + ccc*6 + cc*8 + c*16 + none*0) / total_combinations

print("average payout:")
print(average)

# exercise 2: part (b)
oddswin = 48 / 64
print("odds of winning:")
print(oddswin)

# simulate game


# def simulate_game(starting_coins, payout_average, win_probability, trials):
#     results = []
#     for _ in range(trials):
#         coins = starting_coins
#         plays = 0
#         while coins > 0:
#             plays += 1
#             coins -= 1
#             if random.random() < win_probability:
#                 coins += payout_average
#         results.append(plays)
#     return results

# starting_coins, payout_average, win_probability, trials = 10, 4.53, 0.75, 100
# results = simulate_game(starting_coins, payout_average, win_probability, trials)

# mean_plays = sum(results) / len(results)
# median_plays = sorted(results)[len(results) // 2]

# print("mean plays:")
# print(f"{mean_plays:.2f}")
# print("median plays:")
# print(f"{median_plays:.2f}")


# exercise 3: part (a)

def odds_same_birthday(n):
    return 1 - math.prod([(365 - i) / 365 for i in range(n)])

print("odds of two people having the same birthday:")
print(f"{odds_same_birthday(2):.15f}")

# exercise 3: part (b)

def odds_birthday_over_50():
    for i in range (10, 50):
        if odds_same_birthday(i) >= 0.5:
            return i, odds_same_birthday(i)
    else:
        return False
    
print("number of people needed for a 50% chance of two people having the same birthday:")
print(odds_birthday_over_50())

# exercise 3: part (2a)

 # we fill each slot of the hashset with a random birthday, and check if the slot is already filled
def simulate_group_size():
    """Simulate the size of the group required to cover all 365 days."""
    days_covered = set()
    group_size = 0
    while len(days_covered) < 365:
        new_birthday = random.randint(1, 365)  # Randomly select a birthday
        days_covered.add(new_birthday)  # Add it to the set of covered days
        group_size += 1
    return group_size

# pretty much the same simulation as above, but with a different return value
def average_group_size(trials=1000):
    """Calculate the average group size over a number of trials."""
    total_group_size = sum(simulate_group_size() for _ in range(trials))
    return total_group_size / trials

# Run the simulation
trials = 1000
result = average_group_size(trials)

print(f"Expected group size to cover all days: {result:.2f} (based on {trials} trials)")

# tadaa, great success!
