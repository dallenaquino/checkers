def board_positions():
    for i in range(8):
        for j in range((i + 1) % 2, 8, 2):
            yield i, j
