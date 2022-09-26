fitness = [8, 12, 27, 4, 45, 17]
sum_fitness = sum(fitness)
relative_portion = [None for _ in fitness]

for idx, fit in enumerate(fitness):
    relative_portion[idx] = fit / sum_fitness * 100

for portion in relative_portion:
    print(f"{portion:.2f} ", end="")
