import random

#Ignore this file

uma = [15,5,-5,-15]

e1_sum = 0
for i in range(100000):
    score = [25000,25000,25000,25000]
    

    for i in range(7):
        a = random.sample(range(4), 1)[0]
        b = random.sample(range(4), 1)[0]
        amount = random.randint(1000, 8000)
        score[a] += amount
        score[b] -= amount

    bigger = sum(1 for x in score[1:] if x > score[0])
    final_points = uma[bigger] + score[0] / 1000 - 25
    e1_sum += final_points

s4_sum = 0
for i in range(100000):
    score = [20500,25000,25000,25000]
    for i in range(1):
        a, b = random.sample(range(4), 2)
        amount = random.randint(1000, 8000)
        score[a] += amount
        score[b] -= amount
    bigger = sum(1 for x in score[1:] if x > score[0])
    final_points = uma[bigger] + score[0] / 1000 - 25
    s4_sum += final_points

print("E1 average:", e1_sum / 100000)
print("S4 average:", s4_sum / 100000)