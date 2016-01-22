def hCostToorac2(path, dim):
    state = path[-1]
    cost = 0

    for i, num in enumerate(state):
        if num == 0: continue
        cur_row = i // dim[0]
        cur_col = i % dim[0]
        should_row = (num-1) // dim[0]
        should_col = (num-1) % dim[0]
        if cur_row != should_row:
            cost += 1
        if cur_col != should_col:
            cost += 1
    return cost

print(hCostToorac2(["",[ 1, 2, 3, 4,
                         5, 6, 7, 8,
                         9,10,11,12,
                        13,14,15, 0]], (4,4)))
