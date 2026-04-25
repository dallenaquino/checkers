import itertools
from typing import Optional

import pygame
from pygame._sdl2.video import Window

from CheckersGame import CheckerboardCodes, CheckersMove, PushMove, JumpMove, PromotionMove, CheckersMoveInfo, \
    CheckersGame
from GameEngine import GameEngine


class CheckerPieceSprite(pygame.sprite.Sprite):
    RED_PIECE = pygame.image.load("images/red_piece.png")
    RED_KING = pygame.image.load("images/red_king.png")
    BLACK_PIECE = pygame.image.load("images/black_piece.png")
    BLACK_KING = pygame.image.load("images/black_king.png")

    def __init__(self, piece_type: CheckerboardCodes, pos=(0, 0), size=None):
        super().__init__()
        self.original_image = self._get_image_from_type(piece_type)
        self.image = self.original_image
        if size is not None:
            self.rescale(size)
        self.rect = self.image.get_rect(center=pos)
        self.type = piece_type

    def rescale(self, new_size: tuple[float, float], new_pos=None):
        self.image = pygame.transform.scale(self.original_image, new_size)
        if new_pos is not None:
            self.rect = self.image.get_rect(center=new_pos)

    def move(self, pos_vec: pygame.math.Vector2):
        self.rect = self.image.get_rect(center = (pos_vec.x, pos_vec.y))

    @classmethod
    def _get_image_from_type(cls, piece_type):
        if piece_type == CheckerboardCodes.BLACK_PIECE:
            return cls.BLACK_PIECE.convert_alpha()
        if piece_type == CheckerboardCodes.BLACK_KING:
            return cls.BLACK_KING.convert_alpha()
        if piece_type == CheckerboardCodes.RED_PIECE:
            return cls.RED_PIECE.convert_alpha()
        if piece_type == CheckerboardCodes.RED_KING:
            return cls.RED_KING.convert_alpha()
        raise Exception(f"Invalid piece type: {piece_type}")


class SkipIteration(Exception):
    """Used in place of \"continue\" so that whatever comes after
    a try block will be executed, rather than skipped"""
    pass


class CheckersUI:
    
    def __init__(self, game_engine):
        self._game_engine: CheckersGame = game_engine

        self._board_image = None
        self._original_board_width = None
        self._original_board_height = None
        self._board_surface = None
        self._original_border_width = None
        self._original_square_width = None
        self._original_piece_width = None
        self._board_pos = None
        self._sprite_grid: list[list[Optional[CheckerPieceSprite]]] = None
        self._curr_scale = None
        self._sprite_group = None
        self._window = None

    def run(self, size=(800, 400)):
        pygame.init()
        self._window = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption("Checkers")

        self._board_image = pygame.image.load('images/board.png').convert_alpha()
        self._original_board_width, self._original_board_height = self._board_image.get_size()
        self._board_surface = self._board_image
        self._original_border_width = 32
        self._original_square_width = (self._original_board_width - 2 * self._original_border_width) / 8
        self._original_piece_width = self._original_square_width * 0.9
        self._board_pos = (0, 0)
        self._sprite_grid: list[list[Optional[CheckerPieceSprite]]] = [[None for _ in range(8)] for _ in range(8)]
        self._curr_scale = 1
        self._sprite_group = pygame.sprite.Group()
        self._create_sprites(self._original_piece_width)
        pygame.display.flip()
        sdl_window = Window.from_display_module()
        sdl_window.maximize()

        self.resize_board(*self._window.get_size())
        pygame.display.flip()
        clock = pygame.time.Clock()
        in_motion = False
        sprite_positions = None
        running = True
        printed_winner = False
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    running = False
                    break
                if e.type == pygame.VIDEORESIZE:
                    self.resize_board(*e.size)
            if not running:
                break

            if self._game_engine.current_state.is_terminal() and not in_motion:
                # TODO: display winner
                if not printed_winner:
                    print(f'Winner: {self._game_engine.current_state.get_winner()}')
                    printed_winner = True
                continue

            print(self._game_engine.current_state.moves_since_capture)
            try:
                if not in_motion:
                    # A new move can be made
                    move_info: CheckersMoveInfo = self._game_engine.get_next_move()
                    self._game_engine.make_move(move_info)
                    move = move_info.move
                    r, c = move.starting_coords
                    if self._sprite_grid[r][c] is None:
                        raise SkipIteration
                    sprite_positions = self.compute_sprite_positions(move)
                    in_motion = True
                if in_motion:
                    # Start/continue moving the sprite
                    r, c = move.starting_coords
                    sprite = self._sprite_grid[r][c]
                    try:
                        v = next(sprite_positions)
                        sprite.move(v)
                    except StopIteration as e:
                        ending_r, ending_c = e.value
                        in_motion = False
                        self._sprite_grid[ending_r][ending_c] = sprite
                        self._sprite_grid[r][c] = None
                        if isinstance(move, PromotionMove):
                            # Change sprite image to king
                            new_type = CheckerboardCodes.promote(sprite.type)
                            new_sprite = CheckerPieceSprite(new_type, sprite.rect.center, sprite.rect.size)
                            self._sprite_grid[ending_r][ending_c] = new_sprite
                            self._sprite_group.add(new_sprite)
                            sprite.kill()


            except SkipIteration:
                pass
            self._window.blit(self._board_surface, self._board_pos)
            self._sprite_group.draw(self._window)
            pygame.display.flip()
            clock.tick(60)

        pygame.quit()

    def compute_sprite_positions(self, move, num_steps=20):
        unit_vector = pygame.math.Vector2(self._pos_from_grid_coord(0, 0)) - pygame.math.Vector2(self._pos_from_grid_coord(1, 1))

        def divide_distance(c1, c2):
            p1 = pygame.math.Vector2(self._pos_from_grid_coord(*c1))
            p2 = pygame.math.Vector2(self._pos_from_grid_coord(*c2))
            direction = p2 - p1
            n_units = round(direction.length() / unit_vector.length())
            total_steps = num_steps * n_units
            step = direction / total_steps
            for i in range(total_steps):
                yield p1 + (i + 1) * step

        if isinstance(move, PushMove):
            ending_coord = move.compute_next_coord(move.starting_coords, move.action)
            yield from divide_distance(move.starting_coords, ending_coord)
            return ending_coord
        elif isinstance(move, JumpMove):
            coord2 = move.starting_coords
            for i, action in enumerate(move.actions):
                coord1 = coord2
                coord2 = move.compute_next_coord(coord1, action)
                yield from divide_distance(coord1, coord2)
                jumped_x, jumped_y = ((x + y) // 2 for x, y in zip(coord1, coord2))
                self._sprite_grid[jumped_x][jumped_y].kill()
                self._sprite_grid[jumped_x][jumped_y] = None
            return coord2
        return move.starting_coords

    def resize_piece(self, r, c):
        new_sprite_width = self._curr_scale * self._original_piece_width
        self._sprite_grid[r][c].rescale((new_sprite_width, new_sprite_width), self._pos_from_grid_coord(r, c))


    def resize_board(self, window_width, window_height):
        scale = min(window_width / self._original_board_width, window_height / self._original_board_height)
        self._curr_scale = scale
        self._board_surface = pygame.transform.scale_by(self._board_image, scale)
        new_width, new_height = self._board_surface.get_size()
        self._board_pos = (window_width - new_width) / 2, (window_height - new_height) / 2
        self._window.blit(self._board_surface, self._board_pos)
        for i, j in filter(lambda coord: self._sprite_grid[coord[0]][coord[1]] is not None,
                           itertools.permutations(range(8), 2)):
            self.resize_piece(i, j)
        self._sprite_group.draw(self._window)

    def _create_sprites(self, width):
        for i in range(3):
            for j in range((i + 1) % 2, 8, 2):
                black_sprite = CheckerPieceSprite(CheckerboardCodes.BLACK_PIECE, self._pos_from_grid_coord(i, j), (width, width))
                self._sprite_grid[i][j] = black_sprite
                red_sprite = CheckerPieceSprite(CheckerboardCodes.RED_PIECE, self._pos_from_grid_coord(-i - 1, -j - 1), (width, width))
                self._sprite_grid[-i - 1][-j - 1] = red_sprite
                self._sprite_group.add(black_sprite, red_sprite)

    def _pos_from_grid_coord(self, r, c):
        scale = self._curr_scale
        border_width = scale * self._original_border_width
        square_width = scale * self._original_square_width
        x = ((c % 8) + 0.5) * square_width + border_width + self._board_pos[0]
        y = ((r % 8) + 0.5) * square_width + border_width + self._board_pos[1]
        return x, y

    def _grid_coord_from_pos(self, pos):
        scale = self._curr_scale
        border_width = scale * self._original_border_width
        square_width = scale * self._original_square_width
        c, r = (round((pos[i] - self._board_pos[i] - border_width) / square_width - 0.5) % 8 for i in range(2))
        return c, r