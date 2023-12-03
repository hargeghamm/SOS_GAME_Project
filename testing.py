import itertools
from game_board import GameBoard
from players import AbstractPlayer, PlayerType
from game import Game
import matplotlib.pyplot as plt
from collections import defaultdict

def run_game(player1_type, player2_type):
    game_board = GameBoard(size=5)  
    player1_kwargs = {"depth": 3} if player1_type == PlayerType.MINIMAX_PLAYER_Alpha_Beta else {}
    player2_kwargs = {"depth": 3} if player2_type == PlayerType.MINIMAX_PLAYER_Alpha_Beta else {}

    player1 = AbstractPlayer.create(player_type=player1_type, **player1_kwargs)
    player2 = AbstractPlayer.create(player_type=player2_type, **player2_kwargs)
    game = Game(player1, player2, game_board)
    game.run()
    return game.get_results()


def main():
    player_types = [PlayerType.RANDOM_PLAYER, PlayerType.MINIMAX_PLAYER_Alpha_Beta, PlayerType.MONTE_CARLO_PLAYER, PlayerType.GREEDY_PLAYER]
    results = []

    for _ in range(2):
        for player1_type, player2_type in itertools.product(player_types, repeat=2):
            result = run_game(player1_type, player2_type)
            results.append(result)

    # Process and display results
    for result in results:
        print(f"Player 1 (Type: {result['player_1_type']}) Score: {result['player_1_score']}, Player 2 (Type: {result['player_2_type']}) Score: {result['player_2_score']}, Winner: {result['winner']}, Tie: {result['tie']}")

        
    win_count, loss_count = process_results(results)
    plot_results(win_count, loss_count)


def process_results(results):
    win_count = defaultdict(int)
    loss_count = defaultdict(int)

    for result in results:
        player1_type = str(result['player_1_type'])
        player2_type = str(result['player_2_type'])
        winner = result['winner']
        if winner == 'Player 1':
            win_count[player1_type] += 1
            loss_count[player2_type] += 1
        elif winner == 'Player 2':
            win_count[player2_type] += 1
            loss_count[player1_type] += 1

    return win_count, loss_count

def plot_results(win_count, loss_count):
    labels = list(win_count.keys())
    win_values = [win_count[label] for label in labels]
    loss_values = [loss_count[label] for label in labels]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar(x, win_values, width, label='Wins')
    ax.bar(x, loss_values, width, bottom=win_values, label='Losses')

    ax.set_ylabel('Counts')
    ax.set_title('Wins and Losses by Player Type')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.show()

if __name__ == "__main__":
    main()
