import itertools
import math
import random
from abc import ABC, abstractmethod
from enum import Enum

from game_board import GameBoard
from location import Location
from move import Move
from sign import Sign
from typing import List

MIN_AGENT = 1
MAX_AGENT = 0


class PlayerType(Enum):
    USER_PLAYER = 1
    RANDOM_PLAYER = 2
    MINIMAX_PLAYER_Alpha_Beta = 3
    MONTE_CARLO_PLAYER = 4
    GREEDY_PLAYER = 5


class AbstractPlayer(ABC):
    _score: int = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def make_move(self, game_board: GameBoard) -> Move:
        raise NotImplementedError

    @abstractmethod
    def get_type(self) -> PlayerType:
        raise NotImplementedError

    def update_score(self, score: int) -> None:
        self._score += score

    def get_score(self) -> int:
        return self._score

    @staticmethod
    def create(player_type: PlayerType, **kwargs):
        # if player_type == PlayerType.USER_PLAYER:
        #     return UserPlayer(**kwargs)
        if player_type == PlayerType.RANDOM_PLAYER:
            return RandomPlayer(**kwargs)
        if player_type == PlayerType.MINIMAX_PLAYER_Alpha_Beta:
            return MiniMax_AlphaBeta_Player(**kwargs)
        if player_type == PlayerType.MONTE_CARLO_PLAYER:
            return MonteCarloPlayer(**kwargs)
        if player_type == PlayerType.GREEDY_PLAYER:
            return GreedyPlayer(**kwargs)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(score={self.get_score()})"


class RandomPlayer(AbstractPlayer):
    def get_type(self) -> PlayerType:
        return PlayerType.RANDOM_PLAYER

    def make_move(self, game_board: GameBoard) -> Move:
        empty_locations: list[Location] = game_board.get_empty_locations()
        location: Location = random.choice(empty_locations)
        sign: Sign = Sign(random.choice(Sign.get_input_valid_signs()))
        return Move(location, sign)


# class UserPlayer(AbstractPlayer):
#     def get_type(self) -> PlayerType:
#         return PlayerType.USER_PLAYER

#     def make_move(self, game_board: GameBoard) -> Move:
#         while True:
#             try:
#                 x: int = int(
#                     input(
#                         f"Enter x (rows) coordinate position 0 - {game_board.get_size() - 1}: "
#                     )
#                 )
#                 y: int = int(
#                     input(
#                         f"Enter y (columns) coordinate position 0- {game_board.get_size() - 1}: "
#                     )
#                 )
#                 sign: Sign = Sign.from_user_input(
#                     (input("Enter letter S or O: ").upper())
#                 )
#             except ValueError as e:
#                 print(str(e))
#                 continue

#             move: Move = Move(Location(x, y), sign)
#             if not game_board.move_is_ok(move):
#                 print("The move is not valid! Try again.\n")
#                 continue

#             return move


class MiniMax_AlphaBeta_Player(AbstractPlayer):
    def __init__(self, *args, depth: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._depth = depth

    def get_type(self) -> PlayerType:
        return PlayerType.MINIMAX_PLAYER_Alpha_Beta

    def _count_pruned_nodes_percentage(
        self, game_board: GameBoard, skipped_node_count: int
    ) -> float:
        empty_locations = len(game_board.get_empty_locations())
        total_nodes = math.prod(
            map(
                lambda x: x * 2,
                range(empty_locations - self._depth + 1, empty_locations + 1),
            )
        )

        return 100 * skipped_node_count / total_nodes if total_nodes else 0

    def make_move(self, game_board: GameBoard) -> Move:
        best_move = Move(Location(0, 0), Sign.EMPTY)
        best_score = float("-inf")

        scores = []
        skipped_node_count = [0]
        for location, sign in itertools.product(
            game_board.get_empty_locations(), Sign.get_input_valid_signs()
        ):
            game_board.play_move(Move(location, sign))
            current_score = self._minimax_alpha_beta(
                game_board, self._depth, False, skipped_node_count
            )

            scores.append(current_score)
            game_board.play_move(Move(location, Sign.EMPTY))
            if current_score > best_score:
                best_score = current_score
                best_move = Move(location, sign)

        if best_score == min(scores):
            best_move = self._smart_random_move(game_board)

        print(
            f"Alpha-Beta pruned {self._count_pruned_nodes_percentage(game_board, skipped_node_count[0])}% of the nodes."
        )

        return best_move

    def _random_move(self, game_board: GameBoard) -> Move:
        return RandomPlayer().make_move(game_board)

    def _block_evaluation(self, game_board: GameBoard) -> float:
        score = 1
        for location in game_board.get_locations_with_sign(Sign.LETTER_S):
            if game_board.is_almost_sos(location):
                score += 1

        return 1 / score

    def _smart_random_move(self, game_board: GameBoard) -> Move:
        best_moves = []
        for location, sign in itertools.product(
            game_board.get_empty_locations(), Sign.get_input_valid_signs()
        ):
            game_board.play_move(Move(location, sign))
            score = self._block_evaluation(game_board)
            if score == 1:
                best_moves.append(Move(location, sign))
            game_board.play_move(Move(location, Sign.EMPTY))

        return (
            random.choice(best_moves) if best_moves else self._random_move(game_board)
        )

    def _simple_score_evaluation(self, game_board: GameBoard) -> int:
        return game_board.get_sos_count()

    def _minimax_alpha_beta(
        self,
        game_board: GameBoard,
        depth: int,
        is_max_turn: bool,
        skipped_node_count: list,
        alpha: float = float("-inf"),
        beta: float = float("inf"),
    ) -> float:

        if depth == 0 or not game_board.has_empty_locations():
            return self._simple_score_evaluation(game_board)

        if is_max_turn:
            best_score = float("-inf")
            for location, sign in itertools.product(
                game_board.get_empty_locations(), Sign.get_input_valid_signs()
            ):
                sos_count = game_board.get_sos_count()
                game_board.play_move(Move(location, sign))
                new_sos_count = game_board.get_sos_count()
                diff = new_sos_count - sos_count
                if diff == 0:
                    is_max_turn = not is_max_turn
                score = self._minimax_alpha_beta(
                    game_board, depth - 1, is_max_turn, skipped_node_count, alpha, beta
                )
                game_board.play_move(Move(location, Sign.EMPTY))
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    skipped_node_count[0] += 1
                    break

            return best_score
        else:
            best_score = float("inf")
            for location, sign in itertools.product(
                game_board.get_empty_locations(), Sign.get_input_valid_signs()
            ):
                sos_count = game_board.get_sos_count()
                game_board.play_move(Move(location, sign))
                new_sos_count = game_board.get_sos_count()
                diff = new_sos_count - sos_count
                if diff == 0:
                    is_max_turn = not is_max_turn
                score = self._minimax_alpha_beta(
                    game_board, depth - 1, is_max_turn, skipped_node_count, alpha, beta
                )
                game_board.play_move(Move(location, Sign.EMPTY))
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    skipped_node_count[0] += 1
                    break
            return best_score
        


class MonteCarloPlayer(AbstractPlayer):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_type(self) -> PlayerType:
        return PlayerType.MONTE_CARLO_PLAYER

    def make_move(self, game_board: GameBoard) -> Move:
        return self.select_move(game_board)

    def select_move(self, game_board: GameBoard) -> Move:
        best_move = None
        best_score = float('-inf')

        for move in self.get_possible_moves(game_board):
            score = self.simulate_move(move, game_board)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def simulate_move(self, move: Move, game_board: GameBoard) -> int:
        total_score = 0
        num_simulations = 100  # Number of simulations per move

        for _ in range(num_simulations):
            simulation_score = self.simulate_random_game_from_move(game_board, move)
            total_score += simulation_score

        return total_score

    def simulate_random_game_from_move(self, game_board: GameBoard, move: Move) -> int:
        temp_board = game_board.clone()
        temp_board.apply_move(move)

        while not temp_board.is_full():
            possible_moves = self.get_possible_moves(temp_board)
            if not possible_moves:  # Check if there are no possible moves
                break
            random_move = random.choice(possible_moves)
            temp_board.apply_move(random_move)

        return temp_board.evaluate()

    def get_possible_moves(self, game_board: GameBoard) -> List[Move]:
        possible_moves = []
        for x in range(game_board.get_size()):
            for y in range(game_board.get_size()):
                if game_board.get_location_sign(Location(x, y)) == Sign.EMPTY:
                    for sign in [Sign.LETTER_S, Sign.LETTER_O]:
                        possible_moves.append(Move(Location(x, y), sign))
        return possible_moves

class GreedyPlayer(AbstractPlayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_type(self) -> PlayerType:
        return PlayerType.GREEDY_PLAYER

    def make_move(self, game_board: GameBoard) -> Move:
        return self.select_move(game_board)

    def select_move(self, game_board: GameBoard) -> Move:
        # First check for immediate SOS creation
        for move in self.get_possible_moves(game_board):
            if game_board.is_sos_created_by_move(move):
                return move

        # If no immediate SOS, choose the first available spot
        return self.get_possible_moves(game_board)[0]

    def get_possible_moves(self, game_board: GameBoard) -> List[Move]:
        possible_moves = []
        for x in range(game_board.get_size()):
            for y in range(game_board.get_size()):
                if game_board.get_location_sign(Location(x, y)) == Sign.EMPTY:
                    for sign in [Sign.LETTER_S, Sign.LETTER_O]:
                        possible_moves.append(Move(Location(x, y), sign))
        return possible_moves
