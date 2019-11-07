# -*- coding: utf-8 -*-
#
# This file is part of the python-chess library.
# Copyright (C) 2016-2019 Niklas Fiekas <niklas.fiekas@backscattering.de>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

import chess
import copy
import itertools

from typing import Dict, Generic, Hashable, Iterable, Iterator, List, Optional, Type, TypeVar, Union


class SuicideBoard(chess.Board):

    aliases = ["Suicide", "Suicide chess"]
    uci_variant = "suicide"
    xboard_variant = "suicide"

    tbw_suffix = ".stbw"
    tbz_suffix = ".stbz"
    tbw_magic = b"\x7b\xf6\x93\x15"
    tbz_magic = b"\xe4\xcf\xe7\x23"
    pawnless_tbw_suffix = ".gtbw"
    pawnless_tbz_suffix = ".gtbz"
    pawnless_tbw_magic = b"\xbc\x55\xbc\x21"
    pawnless_tbz_magic = b"\xd6\xf5\x1b\x50"
    connected_kings = True
    one_king = False
    captures_compulsory = True

    def pin_mask(self, color: chess.Color, square: chess.Square) -> chess.Bitboard:
        return chess.BB_ALL

    def _attacked_for_king(self, path: chess.Bitboard, occupied: chess.Bitboard) -> bool:
        return False

    def is_check(self) -> bool:
        return False

    def is_into_check(self, move: chess.Move) -> bool:
        return False

    def was_into_check(self) -> bool:
        return False

    def _material_balance(self) -> int:
        return (chess.popcount(self.occupied_co[self.turn]) -
                chess.popcount(self.occupied_co[not self.turn]))

    def is_variant_end(self) -> bool:
        return not all(has_pieces for has_pieces in self.occupied_co)

    def is_variant_win(self) -> bool:
        if not self.occupied_co[self.turn]:
            return True
        else:
            return self.is_stalemate() and self._material_balance() < 0

    def is_variant_loss(self) -> bool:
        if not self.occupied_co[self.turn]:
            return False
        else:
            return self.is_stalemate() and self._material_balance() > 0

    def is_variant_draw(self) -> bool:
        if not self.occupied_co[self.turn]:
            return False
        else:
            return self.is_stalemate() and self._material_balance() == 0

    def has_insufficient_material(self, color: chess.Color) -> bool:
        if self.occupied != self.bishops:
            return False

        # In a position with only bishops, check if all our bishops can be
        # captured.
        we_some_on_light = bool(self.occupied_co[color] & chess.BB_LIGHT_SQUARES)
        we_some_on_dark = bool(self.occupied_co[color] & chess.BB_DARK_SQUARES)
        they_all_on_dark = not (self.occupied_co[not color] & chess.BB_LIGHT_SQUARES)
        they_all_on_light = not (self.occupied_co[not color] & chess.BB_DARK_SQUARES)
        return (we_some_on_light and they_all_on_dark) or (we_some_on_dark and they_all_on_light)

    def generate_pseudo_legal_moves(self, from_mask: chess.Bitboard = chess.BB_ALL, to_mask: chess.Bitboard = chess.BB_ALL) -> Iterator[chess.Move]:
        for move in super().generate_pseudo_legal_moves(from_mask, to_mask):
            # Add king promotions.
            if move.promotion == chess.QUEEN:
                yield chess.Move(move.from_square, move.to_square, chess.KING)

            yield move

    def generate_legal_moves(self, from_mask: chess.Bitboard = chess.BB_ALL, to_mask: chess.Bitboard = chess.BB_ALL) -> Iterator[chess.Move]:
        if self.is_variant_end():
            return

        # Generate captures first.
        found_capture = False
        for move in self.generate_pseudo_legal_captures():
            if chess.BB_SQUARES[move.from_square] & from_mask and chess.BB_SQUARES[move.to_square] & to_mask:
                yield move
            found_capture = True

        # Captures are mandatory. Stop here if any were found.
        if not found_capture:
            not_them = to_mask & ~self.occupied_co[not self.turn]
            for move in self.generate_pseudo_legal_moves(from_mask, not_them):
                if not self.is_en_passant(move):
                    yield move

    def is_legal(self, move: chess.Move) -> bool:
        if not super().is_legal(move):
            return False

        if self.is_capture(move):
            return True
        else:
            return not any(self.generate_pseudo_legal_captures())

    def _transposition_key(self) -> Hashable:
        if self.has_chess960_castling_rights():
            return (super()._transposition_key(), self.kings & self.promoted)
        else:
            return super()._transposition_key()

    def board_fen(self, promoted: Optional[bool] = None) -> str:
        if promoted is None:
            promoted = self.has_chess960_castling_rights()
        return super().board_fen(promoted=promoted)

    def status(self) -> chess.Status:
        status = super().status()
        status &= ~chess.STATUS_NO_WHITE_KING
        status &= ~chess.STATUS_NO_BLACK_KING
        status &= ~chess.STATUS_TOO_MANY_KINGS
        status &= ~chess.STATUS_OPPOSITE_CHECK
        return status


class GiveawayBoard(SuicideBoard):

    aliases = ["Giveaway", "Giveaway chess", "Anti", "Antichess", "Anti chess"]
    uci_variant = "giveaway"
    xboard_variant = "giveaway"
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"

    tbw_suffix = ".gtbw"
    tbz_suffix = ".gtbz"
    tbw_magic = b"\xbc\x55\xbc\x21"
    tbz_magic = b"\xd6\xf5\x1b\x50"
    pawnless_tbw_suffix = ".stbw"
    pawnless_tbz_suffix = ".stbz"
    pawnless_tbw_magic = b"\x7b\xf6\x93\x15"
    pawnless_tbz_magic = b"\xe4\xcf\xe7\x23"

    def __init__(self, fen: Optional[str] = starting_fen, chess960: bool = False) -> None:
        super().__init__(fen, chess960=chess960)

    def reset(self) -> None:
        super().reset()
        self.castling_rights = chess.BB_EMPTY

    def is_variant_win(self) -> bool:
        return not self.occupied_co[self.turn] or self.is_stalemate()

    def is_variant_loss(self) -> bool:
        return False

    def is_variant_draw(self) -> bool:
        return False


class AtomicBoard(chess.Board):

    aliases = ["Atomic", "Atom", "Atomic chess"]
    uci_variant = "atomic"
    xboard_variant = "atomic"

    tbw_suffix = ".atbw"
    tbz_suffix = ".atbz"
    tbw_magic = b"\x55\x8d\xa4\x49"
    tbz_magic = b"\x91\xa9\x5e\xeb"
    connected_kings = True
    one_king = True

    def is_variant_end(self) -> bool:
        return not all(self.kings & side for side in self.occupied_co)

    def is_variant_win(self) -> bool:
        return bool(self.kings and not self.kings & self.occupied_co[not self.turn])

    def is_variant_loss(self) -> bool:
        return bool(self.kings and not self.kings & self.occupied_co[self.turn])

    def has_insufficient_material(self, color: chess.Color) -> bool:
        # Remaining material does not matter if opponent's king is already
        # exploded.
        if not (self.occupied_co[not color] & self.kings):
            return False

        # Bare king can not mate.
        if not (self.occupied_co[color] & ~self.kings):
            return True

        # As long as the opponent's king is not alone, there is always a chance
        # their own pieces explode next to it.
        if self.occupied_co[not color] & ~self.kings:
            # Unless there are only bishops that cannot explode each other.
            if self.occupied == self.bishops | self.kings:
                if not (self.bishops & self.occupied_co[chess.WHITE] & chess.BB_DARK_SQUARES):
                    return not (self.bishops & self.occupied_co[chess.BLACK] & chess.BB_LIGHT_SQUARES)
                if not (self.bishops & self.occupied_co[chess.WHITE] & chess.BB_LIGHT_SQUARES):
                    return not (self.bishops & self.occupied_co[chess.BLACK] & chess.BB_DARK_SQUARES)
            return False

        # Queen or pawn (future queen) can give mate against bare king.
        if self.queens or self.pawns:
            return False

        # Single knight, bishop or rook cannot mate against bare king.
        if chess.popcount(self.knights | self.bishops | self.rooks) == 1:
            return True

        # Two knights cannot mate against bare king.
        if self.occupied == self.knights | self.kings:
            return chess.popcount(self.knights) <= 2

        return False

    def _attacked_for_king(self, path: chess.Bitboard, occupied: chess.Bitboard) -> bool:
        # Can castle onto attacked squares if they are connected to the
        # enemy king.
        enemy_kings = self.kings & self.occupied_co[not self.turn]
        for enemy_king in chess.scan_forward(enemy_kings):
            path &= ~chess.BB_KING_ATTACKS[enemy_king]

        return super()._attacked_for_king(path, occupied)

    def _kings_connected(self) -> bool:
        white_kings = self.kings & self.occupied_co[chess.WHITE]
        black_kings = self.kings & self.occupied_co[chess.BLACK]
        return any(chess.BB_KING_ATTACKS[sq] & black_kings for sq in chess.scan_forward(white_kings))

    def _push_capture(self, move: chess.Move, capture_square: chess.Square, piece_type: chess.PieceType, was_promoted: bool) -> None:
        # Explode the capturing piece.
        self._remove_piece_at(move.to_square)

        # Explode all non pawns around.
        explosion_radius = chess.BB_KING_ATTACKS[move.to_square] & ~self.pawns
        for explosion in chess.scan_forward(explosion_radius):
            self._remove_piece_at(explosion)

        # Destroy castling rights.
        self.castling_rights &= ~explosion_radius

    def is_check(self) -> bool:
        return not self._kings_connected() and super().is_check()

    def was_into_check(self) -> bool:
        return not self._kings_connected() and super().was_into_check()

    def is_into_check(self, move: chess.Move) -> bool:
        self.push(move)
        was_into_check = self.was_into_check()
        self.pop()
        return was_into_check

    def is_legal(self, move: chess.Move) -> bool:
        if self.is_variant_end():
            return False

        if not self.is_pseudo_legal(move):
            return False

        self.push(move)
        legal = bool(self.kings) and not self.is_variant_win() and (self.is_variant_loss() or not self.was_into_check())
        self.pop()

        return legal

    def is_stalemate(self) -> bool:
        return not self.is_variant_loss() and super().is_stalemate()

    def generate_legal_moves(self, from_mask: chess.Bitboard = chess.BB_ALL, to_mask: chess.Bitboard = chess.BB_ALL) -> Iterator[chess.Move]:
        for move in self.generate_pseudo_legal_moves(from_mask, to_mask):
            if self.is_legal(move):
                yield move

    def status(self) -> chess.Status:
        status = super().status()
        status &= ~chess.STATUS_OPPOSITE_CHECK
        if self.turn == chess.WHITE:
            status &= ~chess.STATUS_NO_WHITE_KING
        else:
            status &= ~chess.STATUS_NO_BLACK_KING
        return status


class KingOfTheHillBoard(chess.Board):

    aliases = ["King of the Hill", "KOTH"]
    uci_variant = "kingofthehill"
    xboard_variant = "kingofthehill"  # Unofficial

    tbw_suffix = None
    tbz_suffix = None
    tbw_magic = None
    tbz_magic = None

    def is_variant_end(self) -> bool:
        return bool(self.kings & chess.BB_CENTER)

    def is_variant_win(self) -> bool:
        return bool(self.kings & self.occupied_co[self.turn] & chess.BB_CENTER)

    def is_variant_loss(self) -> bool:
        return bool(self.kings & self.occupied_co[not self.turn] & chess.BB_CENTER)

    def has_insufficient_material(self, color: chess.Color) -> bool:
        return False


class RacingKingsBoard(chess.Board):

    aliases = ["Racing Kings", "Racing", "Race", "racingkings"]
    uci_variant = "racingkings"
    xboard_variant = "racingkings"  # Unofficial
    starting_fen = "8/8/8/8/8/8/krbnNBRK/qrbnNBRQ w - - 0 1"

    tbw_suffix = None
    tbz_suffix = None
    tbw_magic = None
    tbz_magic = None

    def __init__(self, fen: Optional[str] = starting_fen, chess960: bool = False) -> None:
        super().__init__(fen, chess960=chess960)

    def reset(self) -> None:
        self.set_fen(type(self).starting_fen)

    def _gives_check(self, move: chess.Move) -> bool:
        self.push(move)
        gives_check = self.is_check()
        self.pop()
        return gives_check

    def is_legal(self, move: chess.Move) -> bool:
        return super().is_legal(move) and not self._gives_check(move)

    def generate_legal_moves(self, from_mask: chess.Bitboard = chess.BB_ALL, to_mask: chess.Bitboard = chess.BB_ALL) -> Iterator[chess.Move]:
        for move in super().generate_legal_moves(from_mask, to_mask):
            if not self._gives_check(move):
                yield move

    def is_variant_end(self) -> bool:
        if not self.kings & chess.BB_RANK_8:
            return False

        if self.turn == chess.WHITE or self.kings & self.occupied_co[chess.BLACK] & chess.BB_RANK_8:
            return True

        black_kings = self.kings & self.occupied_co[chess.BLACK]
        if not black_kings:
            return True

        black_king = chess.msb(black_kings)

        # White has reached the backrank. The game is over if black can not
        # also reach the backrank on the next move. Check if there are any
        # safe squares for the king.
        targets = chess.BB_KING_ATTACKS[black_king] & chess.BB_RANK_8
        return all(self.attackers_mask(chess.WHITE, target) for target in chess.scan_forward(targets))

    def is_variant_draw(self) -> bool:
        in_goal = self.kings & chess.BB_RANK_8
        return all(in_goal & side for side in self.occupied_co)

    def is_variant_loss(self) -> bool:
        return self.is_variant_end() and not self.kings & self.occupied_co[self.turn] & chess.BB_RANK_8

    def is_variant_win(self) -> bool:
        return self.is_variant_end() and bool(self.kings & self.occupied_co[self.turn] & chess.BB_RANK_8)

    def has_insufficient_material(self, color: chess.Color) -> bool:
        return False

    def status(self) -> chess.Status:
        status = super().status()
        if self.is_check():
            status |= chess.STATUS_RACE_CHECK
        if self.turn == chess.BLACK and all(self.occupied_co[co] & self.kings & chess.BB_RANK_8 for co in chess.COLORS):
            status |= chess.STATUS_RACE_OVER
        if self.pawns:
            status |= chess.STATUS_RACE_MATERIAL
        for color in chess.COLORS:
            if chess.popcount(self.occupied_co[color] & self.knights) > 2:
                status |= chess.STATUS_RACE_MATERIAL
            if chess.popcount(self.occupied_co[color] & self.bishops) > 2:
                status |= chess.STATUS_RACE_MATERIAL
            if chess.popcount(self.occupied_co[color] & self.rooks) > 2:
                status |= chess.STATUS_RACE_MATERIAL
            if chess.popcount(self.occupied_co[color] & self.queens) > 1:
                status |= chess.STATUS_RACE_MATERIAL
        return status


class HordeBoard(chess.Board):

    aliases = ["Horde", "Horde chess"]
    uci_variant = "horde"
    xboard_variant = "horde"  # Unofficial
    starting_fen = "rnbqkbnr/pppppppp/8/1PP2PP1/PPPPPPPP/PPPPPPPP/PPPPPPPP/PPPPPPPP w kq - 0 1"

    tbw_suffix = None
    tbz_suffix = None
    tbw_magic = None
    tbz_magic = None

    def __init__(self, fen: Optional[str] = starting_fen, chess960: bool = False) -> None:
        super().__init__(fen, chess960=chess960)

    def reset(self) -> None:
        self.set_fen(type(self).starting_fen)

    def is_variant_end(self) -> bool:
        return not all(has_pieces for has_pieces in self.occupied_co)

    def is_variant_draw(self) -> bool:
        return not self.occupied

    def is_variant_loss(self) -> bool:
        return bool(self.occupied) and not self.occupied_co[self.turn]

    def is_variant_win(self) -> bool:
        return bool(self.occupied) and not self.occupied_co[not self.turn]

    def has_insufficient_material(self, color: chess.Color) -> bool:
        # TODO: Could detect some cases where the Horde can no longer mate.
        return False

    def status(self) -> chess.Status:
        status = super().status()
        status &= ~chess.STATUS_NO_WHITE_KING

        if chess.popcount(self.occupied_co[chess.WHITE]) <= 36:
            status &= ~chess.STATUS_TOO_MANY_WHITE_PIECES
            status &= ~chess.STATUS_TOO_MANY_WHITE_PAWNS

        if not self.pawns & chess.BB_RANK_8 and not self.occupied_co[chess.BLACK] & self.pawns & chess.BB_RANK_1:
            status &= ~chess.STATUS_PAWNS_ON_BACKRANK

        if self.occupied_co[chess.WHITE] & self.kings:
            status |= chess.STATUS_TOO_MANY_KINGS

        return status


ThreeCheckBoardT = TypeVar("ThreeCheckBoardT", bound="ThreeCheckBoard")

class _ThreeCheckBoardState(Generic[ThreeCheckBoardT], chess._BoardState["ThreeCheckBoardT"]):
    def __init__(self, board: "ThreeCheckBoardT") -> None:
        super().__init__(board)
        self.remaining_checks_w = board.remaining_checks[chess.WHITE]
        self.remaining_checks_b = board.remaining_checks[chess.BLACK]

    def restore(self, board: "ThreeCheckBoardT") -> None:
        super().restore(board)
        board.remaining_checks[chess.WHITE] = self.remaining_checks_w
        board.remaining_checks[chess.BLACK] = self.remaining_checks_b

class ThreeCheckBoard(chess.Board):

    aliases = ["Three-check", "Three check", "Threecheck", "Three check chess", "3-check"]
    uci_variant = "3check"
    xboard_variant = "3check"
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 3+3 0 1"

    tbw_suffix = None
    tbz_suffix = None
    tbw_magic = None
    tbz_magic = None

    def __init__(self, fen: Optional[str] = starting_fen, chess960: bool = False) -> None:
        self.remaining_checks = [3, 3]
        super().__init__(fen, chess960=chess960)

    def reset_board(self) -> None:
        super().reset_board()
        self.remaining_checks[chess.WHITE] = 3
        self.remaining_checks[chess.BLACK] = 3

    def clear_board(self) -> None:
        super().clear_board()
        self.remaining_checks[chess.WHITE] = 3
        self.remaining_checks[chess.BLACK] = 3

    def _board_state(self: ThreeCheckBoardT) -> _ThreeCheckBoardState[ThreeCheckBoardT]:
        return _ThreeCheckBoardState(self)

    def push(self, move: chess.Move) -> None:
        super().push(move)
        if self.is_check():
            self.remaining_checks[not self.turn] -= 1

    def has_insufficient_material(self, color: chess.Color) -> bool:
        # Any remaining piece can give check.
        return not (self.occupied_co[color] & ~self.kings)

    def set_epd(self, epd: str) -> Dict[str, Union[None, str, int, float, chess.Move, List[chess.Move]]]:
        parts = epd.strip().rstrip(";").split(None, 5)

        # Parse ops.
        if len(parts) > 5:
            operations = self._parse_epd_ops(parts.pop(), lambda: type(self)(" ".join(parts) + " 0 1"))
            parts.append(str(operations["hmvc"]) if "hmvc" in operations else "0")
            parts.append(str(operations["fmvn"]) if "fmvn" in operations else "1")
            self.set_fen(" ".join(parts))
            return operations
        else:
            self.set_fen(epd)
            return {}

    def set_fen(self, fen: str) -> None:
        parts = fen.split()

        # Extract check part.
        if len(parts) >= 7 and parts[6][0] == "+":
            check_part = parts.pop(6)
            try:
                w, b = check_part[1:].split("+", 1)
                wc, bc = 3 - int(w), 3 - int(b)
            except ValueError:
                raise ValueError("invalid check part in lichess three-check fen: {}".format(repr(check_part)))
        elif len(parts) >= 5 and "+" in parts[4]:
            check_part = parts.pop(4)
            try:
                w, b = check_part.split("+", 1)
                wc, bc = int(w), int(b)
            except ValueError:
                raise ValueError("invalid check part in three-check fen: {}".format(repr(check_part)))
        else:
            wc, bc = 3, 3

        # Set fen.
        super().set_fen(" ".join(parts))
        self.remaining_checks[chess.WHITE] = wc
        self.remaining_checks[chess.BLACK] = bc

    def epd(self, shredder: bool = False, en_passant: str = "legal", promoted: Optional[bool] = None, **operations: Union[None, str, int, float, chess.Move, Iterable[chess.Move]]) -> str:
        epd = [super().epd(shredder=shredder, en_passant=en_passant, promoted=promoted),
               "{:d}+{:d}".format(max(self.remaining_checks[chess.WHITE], 0),
                                  max(self.remaining_checks[chess.BLACK], 0))]
        if operations:
            epd.append(self._epd_operations(operations))
        return " ".join(epd)

    def is_variant_end(self) -> bool:
        return any(remaining_checks <= 0 for remaining_checks in self.remaining_checks)

    def is_variant_draw(self) -> bool:
        return self.remaining_checks[chess.WHITE] <= 0 and self.remaining_checks[chess.BLACK] <= 0

    def is_variant_loss(self) -> bool:
        return self.remaining_checks[not self.turn] <= 0 < self.remaining_checks[self.turn]

    def is_variant_win(self) -> bool:
        return self.remaining_checks[self.turn] <= 0 < self.remaining_checks[not self.turn]

    def is_irreversible(self, move: chess.Move) -> bool:
        if super().is_irreversible(move):
            return True

        self.push(move)
        gives_check = self.is_check()
        self.pop()
        return gives_check

    def _transposition_key(self) -> Hashable:
        return (super()._transposition_key(),
                self.remaining_checks[chess.WHITE], self.remaining_checks[chess.BLACK])

    def copy(self: ThreeCheckBoardT, stack: Union[bool, int] = True) -> ThreeCheckBoardT:
        board = super().copy(stack=stack)
        board.remaining_checks = self.remaining_checks.copy()
        return board

    def mirror(self: ThreeCheckBoardT) -> ThreeCheckBoardT:
        board = super().mirror()
        board.remaining_checks[chess.WHITE] = self.remaining_checks[chess.BLACK]
        board.remaining_checks[chess.BLACK] = self.remaining_checks[chess.WHITE]
        return board


CrazyhouseBoardT = TypeVar("CrazyhouseBoardT", bound="CrazyhouseBoard")

class _CrazyhouseBoardState(Generic[CrazyhouseBoardT], chess._BoardState["CrazyhouseBoardT"]):
    def __init__(self, board: "CrazyhouseBoardT") -> None:
        super().__init__(board)
        self.pockets_w = board.pockets[chess.WHITE].copy()
        self.pockets_b = board.pockets[chess.BLACK].copy()

    def restore(self, board: "CrazyhouseBoardT") -> None:
        super().restore(board)
        board.pockets[chess.WHITE] = self.pockets_w.copy()
        board.pockets[chess.BLACK] = self.pockets_b.copy()

CrazyhousePocketT = TypeVar("CrazyhousePocketT", bound="CrazyhousePocket")

class CrazyhousePocket:

    def __init__(self, symbols: Iterable[str] = "") -> None:
        self.pieces = {}  # type: Dict[chess.PieceType, int]
        for symbol in symbols:
            self.add(chess.PIECE_SYMBOLS.index(symbol))

    def add(self, pt: chess.PieceType) -> None:
        self.pieces[pt] = self.pieces.get(pt, 0) + 1

    def remove(self, pt: chess.PieceType) -> None:
        self.pieces[pt] -= 1

    def count(self, piece_type: chess.PieceType) -> int:
        return self.pieces.get(piece_type, 0)

    def reset(self) -> None:
        self.pieces.clear()

    def __str__(self) -> str:
        return "".join(chess.piece_symbol(pt) * self.count(pt) for pt in reversed(chess.PIECE_TYPES))

    def __len__(self) -> int:
        return sum(self.pieces.values())

    def __repr__(self) -> str:
        return "CrazyhousePocket('{}')".format(str(self))

    def copy(self: CrazyhousePocketT) -> CrazyhousePocketT:
        pocket = type(self)()
        pocket.pieces = copy.copy(self.pieces)
        return pocket

class CrazyhouseBoard(chess.Board):

    aliases = ["Crazyhouse", "Crazy House", "House", "ZH"]
    uci_variant = "crazyhouse"
    xboard_variant = "crazyhouse"
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR[] w KQkq - 0 1"

    tbw_suffix = None
    tbz_suffix = None
    tbw_magic = None
    tbz_magic = None

    def __init__(self, fen: Optional[str] = starting_fen, chess960: bool = False) -> None:
        self.pockets = [CrazyhousePocket(), CrazyhousePocket()]
        super().__init__(fen, chess960=chess960)

    def reset_board(self) -> None:
        super().reset_board()
        self.pockets[chess.WHITE].reset()
        self.pockets[chess.BLACK].reset()

    def clear_board(self) -> None:
        super().clear_board()
        self.pockets[chess.WHITE].reset()
        self.pockets[chess.BLACK].reset()

    def _board_state(self: CrazyhouseBoardT) -> _CrazyhouseBoardState[CrazyhouseBoardT]:
        return _CrazyhouseBoardState(self)

    def push(self, move: chess.Move) -> None:
        super().push(move)
        if move.drop:
            self.pockets[not self.turn].remove(move.drop)

    def _push_capture(self, move: chess.Move, capture_square: chess.Square, piece_type: chess.PieceType, was_promoted: bool) -> None:
        if was_promoted:
            self.pockets[self.turn].add(chess.PAWN)
        else:
            self.pockets[self.turn].add(piece_type)

    def can_claim_fifty_moves(self) -> bool:
        return False

    def is_seventyfive_moves(self) -> bool:
        return False

    def is_irreversible(self, move: chess.Move) -> bool:
        return self._reduces_castling_rights(move)

    def _transposition_key(self) -> Hashable:
        return (super()._transposition_key(),
                self.promoted,
                str(self.pockets[chess.WHITE]), str(self.pockets[chess.BLACK]))

    def legal_drop_squares_mask(self) -> chess.Bitboard:
        king = self.king(self.turn)
        if king is None:
            return ~self.occupied

        king_attackers = self.attackers_mask(not self.turn, king)

        if not king_attackers:
            return ~self.occupied
        elif chess.popcount(king_attackers) == 1:
            return chess.BB_BETWEEN[king][chess.msb(king_attackers)] & ~self.occupied
        else:
            return chess.BB_EMPTY

    def legal_drop_squares(self) -> chess.SquareSet:
        return chess.SquareSet(self.legal_drop_squares_mask())

    def is_pseudo_legal(self, move: chess.Move) -> bool:
        if move.drop and move.from_square == move.to_square:
            return (
                move.drop != chess.KING and
                not chess.BB_SQUARES[move.to_square] & self.occupied and
                not (move.drop == chess.PAWN and chess.BB_SQUARES[move.to_square] & chess.BB_BACKRANKS) and
                self.pockets[self.turn].count(move.drop) > 0)
        else:
            return super().is_pseudo_legal(move)

    def is_legal(self, move: chess.Move) -> bool:
        if move.drop:
            return self.is_pseudo_legal(move) and bool(self.legal_drop_squares_mask() & chess.BB_SQUARES[move.to_square])
        else:
            return super().is_legal(move)

    def generate_pseudo_legal_drops(self, to_mask: chess.Bitboard = chess.BB_ALL) -> Iterator[chess.Move]:
        for to_square in chess.scan_forward(to_mask & ~self.occupied):
            for pt, count in self.pockets[self.turn].pieces.items():
                if count and (pt != chess.PAWN or not chess.BB_BACKRANKS & chess.BB_SQUARES[to_square]):
                    yield chess.Move(to_square, to_square, drop=pt)

    def generate_legal_drops(self, to_mask: chess.Bitboard = chess.BB_ALL) -> Iterator[chess.Move]:
        return self.generate_pseudo_legal_drops(to_mask=self.legal_drop_squares_mask() & to_mask)

    def generate_legal_moves(self, from_mask: chess.Bitboard = chess.BB_ALL, to_mask: chess.Bitboard = chess.BB_ALL) -> Iterator[chess.Move]:
        return itertools.chain(
            super().generate_legal_moves(from_mask, to_mask),
            self.generate_legal_drops(from_mask & to_mask))

    def parse_san(self, san: str) -> chess.Move:
        if "@" in san:
            uci = san.rstrip("+# ")
            if uci[0] == "@":
                uci = "P" + uci
            move = chess.Move.from_uci(uci)
            if not self.is_legal(move):
                raise ValueError("illegal drop san: {} in {}".format(repr(san), self.fen()))
            return move
        else:
            return super().parse_san(san)

    def has_insufficient_material(self, color: chess.Color) -> bool:
        # In practise no material can leave the game, but this is easy to
        # implement anyway. Note that bishops can be captured and put onto
        # a different color complex.
        return (
            chess.popcount(self.occupied) + sum(len(pocket) for pocket in self.pockets) <= 3 and
            not self.pawns and
            not self.rooks and
            not self.queens and
            not any(pocket.count(chess.PAWN) for pocket in self.pockets) and
            not any(pocket.count(chess.ROOK) for pocket in self.pockets) and
            not any(pocket.count(chess.QUEEN) for pocket in self.pockets))

    def set_fen(self, fen: str) -> None:
        position_part, info_part = fen.split(None, 1)

        # Transform to lichess-style ZH FEN.
        if position_part.endswith("]"):
            if position_part.count("/") != 7:
                raise ValueError("expected 8 rows in position part of zh fen: {}", format(repr(fen)))
            position_part = position_part[:-1].replace("[", "/", 1)

        # Split off pocket part.
        if position_part.count("/") == 8:
            position_part, pocket_part = position_part.rsplit("/", 1)
        else:
            pocket_part = ""

        # Parse pocket.
        white_pocket = CrazyhousePocket(c.lower() for c in pocket_part if c.isupper())
        black_pocket = CrazyhousePocket(c for c in pocket_part if not c.isupper())

        # Set FEN and pockets.
        super().set_fen(position_part + " " + info_part)
        self.pockets[chess.WHITE] = white_pocket
        self.pockets[chess.BLACK] = black_pocket

    def board_fen(self, promoted: Optional[bool] = None) -> str:
        if promoted is None:
            promoted = True
        return super().board_fen(promoted=promoted)

    def epd(self, shredder: bool = False, en_passant: str = "legal", promoted: Optional[bool] = None, **operations: Union[None, str, int, float, chess.Move, Iterable[chess.Move]]) -> str:
        epd = super().epd(shredder=shredder, en_passant=en_passant, promoted=promoted)
        board_part, info_part = epd.split(" ", 1)
        return "{}[{}{}] {}".format(board_part, str(self.pockets[chess.WHITE]).upper(), str(self.pockets[chess.BLACK]), info_part)

    def copy(self: CrazyhouseBoardT, stack: Union[bool, int] = True) -> CrazyhouseBoardT:
        board = super().copy(stack=stack)
        board.pockets[chess.WHITE] = self.pockets[chess.WHITE].copy()
        board.pockets[chess.BLACK] = self.pockets[chess.BLACK].copy()
        return board

    def mirror(self: CrazyhouseBoardT) -> CrazyhouseBoardT:
        board = super().mirror()
        board.pockets[chess.WHITE] = self.pockets[chess.BLACK].copy()
        board.pockets[chess.BLACK] = self.pockets[chess.WHITE].copy()
        return board

    def status(self) -> chess.Status:
        status = super().status()

        if chess.popcount(self.pawns) + self.pockets[chess.WHITE].count(chess.PAWN) + self.pockets[chess.BLACK].count(chess.PAWN) <= 16:
            status &= ~chess.STATUS_TOO_MANY_BLACK_PAWNS
            status &= ~chess.STATUS_TOO_MANY_WHITE_PAWNS

        if chess.popcount(self.occupied) + len(self.pockets[chess.WHITE]) + len(self.pockets[chess.BLACK]) <= 32:
            status &= ~chess.STATUS_TOO_MANY_BLACK_PIECES
            status &= ~chess.STATUS_TOO_MANY_WHITE_PIECES

        return status


VARIANTS = [
    chess.Board,
    SuicideBoard, GiveawayBoard,
    AtomicBoard,
    KingOfTheHillBoard,
    RacingKingsBoard,
    HordeBoard,
    ThreeCheckBoard,
    CrazyhouseBoard,
]  # type: List[Type[chess.Board]]


def find_variant(name: str) -> Type[chess.Board]:
    """Looks for a variant board class by variant name."""
    for variant in VARIANTS:
        if any(alias.lower() == name.lower() for alias in variant.aliases):
            return variant
    raise ValueError("unsupported variant: {}".format(name))
