from __future__ import annotations
from colored import fg

from enum import Enum
color_blue = fg('blue')
color_red = fg('red')
class Sign(str, Enum):
    LETTER_S = (color_blue + "S")
    LETTER_O = (color_red + "O")
    EMPTY = "_"

    @classmethod
    def get_input_valid_signs(cls) -> list[Sign]:
        return [cls.LETTER_S, cls.LETTER_O]

    @classmethod
    def from_user_input(cls, s: str) -> Sign:
        if Sign(s) not in cls.get_input_valid_signs():
            raise ValueError("Invalid input")

        return cls(s)
