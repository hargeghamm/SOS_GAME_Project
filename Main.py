import math

from game import Game
from game_board import GameBoard
from players import AbstractPlayer, PlayerType

SELECT_PLAYER_TEXT = """1. User Player
2. Random Player
3. MiniMax Alpha Beta Player
4. Monte Carlo Player
5. Greedy Player
"""
INPUT_BOARD_SIZE_TEXT = "Input a board size N (NxN): "

def _create_game_board():
    board_size = _input_board_size()
    return GameBoard(size=board_size)

def _select_player_type():
    while True:
        try:
            player_type = int(input(SELECT_PLAYER_TEXT))
            if player_type in range(1, 6):
                return PlayerType(player_type)
            else:
                print("Invalid option. Please select a valid player type.")
        except ValueError as e:
            print(e)

def _input_depth(game_board: GameBoard):
    while True:
        try:
            depth_str = input("Input depth for MiniMax Alpha Beta player (Optional): ")
            if depth_str:
                depth = int(depth_str)
            else:
                depth = int(math.log(game_board.get_size(), 2))

            return depth
        except ValueError as e:
            print(e)
            continue

def _select_player(game_board: GameBoard):
    player_type = _select_player_type()
    player_kwargs = {}
    if player_type == PlayerType.MINIMAX_PLAYER_Alpha_Beta:
        player_kwargs["depth"] = _input_depth(game_board)

    return AbstractPlayer.create(player_type=player_type, **player_kwargs)

def _input_board_size():
    while True:
        try:
            size = int(input(INPUT_BOARD_SIZE_TEXT))
            if size >= 3:
                return size
            else:
                print("Size must be greater than 2.")
        except ValueError as e:
            print(e)
            continue

def main():
    game_board = _create_game_board()
    player_1 = _select_player(game_board)
    player_2 = _select_player(game_board)
    Game(player_1=player_1, player_2=player_2, game_board=game_board).run()

if __name__ == "__main__":
    main()
