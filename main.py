# Imports
import pygame
import pygame.freetype


# Define constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# Pygame setup
pygame.init()
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('Pong via reinforcement learning')
clock = pygame.time.Clock()
GAME_FONT = pygame.freetype.SysFont("sans-serif", 20)


# Config
bat_acceleration = 4
ball_x_vel = 8
ball_y_vel = 8


# Define objects
class Ball:
    def __init__(self, centre_x_pos, centre_y_pos, radius, ball_x_vel, ball_y_vel):
        self.centre_x_pos = centre_x_pos
        self.centre_y_pos = centre_y_pos
        self.radius = radius
        self.color = WHITE
        self.x_vel = ball_x_vel
        self.y_vel = ball_y_vel

    def move(self):
        self.centre_x_pos = self.centre_x_pos + self.x_vel
        self.centre_y_pos = self.centre_y_pos + self.y_vel

    def handle_bounce(self, bot_bat, score):
        left_edge_pos = self.centre_x_pos - self.radius
        right_edge_pos = self.centre_x_pos + self.radius
        top_edge_pos = self.centre_y_pos - self.radius
        bot_edge_pos = self.centre_y_pos + self.radius

        # Handle bounce off of the sides of the screen
        if left_edge_pos <= 0:
            self.x_vel *= -1
            self.centre_x_pos += -left_edge_pos

        elif right_edge_pos >= window_width:
            self.x_vel *= -1
            self.centre_x_pos -= right_edge_pos - window_width

        # (Temporarily) handle bounce off of the top of the screen
        if top_edge_pos <= 0:
            self.y_vel *= -1
            self.centre_y_pos += -top_edge_pos

        # Reset ball at top if it falls off the bottom of the screen, decrease score
        elif bot_edge_pos >= window_height:
            self.centre_y_pos = self.radius
            score.modify(-50)

        # Handle bounce off of bottom bat, increase score
        if (
            bot_edge_pos > bot_bat.top_edge_pos
            and left_edge_pos < bot_bat.right_edge_pos
            and right_edge_pos > bot_bat.left_edge_pos
        ):
            self.y_vel *= -1
            self.centre_y_pos -= bot_edge_pos - bot_bat.top_edge_pos
            score.modify(25)

    def draw(self):
        pygame.draw.circle(window, self.color, (self.centre_x_pos, self.centre_y_pos), self.radius)


class Bat:
    def __init__(self, top_edge_pos, bot_edge_pos, left_edge_pos, right_edge_pos):
        self.top_edge_pos = top_edge_pos
        self.bot_edge_pos = bot_edge_pos
        self.left_edge_pos = left_edge_pos
        self.right_edge_pos = right_edge_pos
        self.acceleration = bat_acceleration
        self.x_velocity = 0

    def accelerate(self, direction):
        if direction == "left":
            self.x_velocity -= self.acceleration
        elif direction == "right":
            self.x_velocity += self.acceleration

    def decelerate(self, direction):
        if direction == "left":
            self.x_velocity += self.acceleration
        elif direction == "right":
            self.x_velocity -= self.acceleration

    def move(self):
        bat_width = self.right_edge_pos - self.left_edge_pos
        self.left_edge_pos += self.x_velocity
        self.right_edge_pos += self.x_velocity
        if self.left_edge_pos < 0:
            self.left_edge_pos = 0
            self.right_edge_pos = bat_width
        elif self.right_edge_pos > window_width:
            self.right_edge_pos = window_width
            self.left_edge_pos = window_width - bat_width

    def draw(self):
        rectangle = pygame.Rect(self.left_edge_pos, self.top_edge_pos,
                                self.right_edge_pos - self.left_edge_pos,
                                self.bot_edge_pos - self.top_edge_pos)
        pygame.draw.rect(window, WHITE, rectangle)


class Score:
    def __init__(self, x_pos, y_pos, value=0):
        self.value = value
        self.x_pos = x_pos
        self.y_pos = y_pos

    def modify(self, amount):
        self.value += amount

    def draw(self):
        GAME_FONT.render_to(window, (self.x_pos, self.y_pos),  "Score: " + str(self.value), WHITE)


# Initial setup
ball = Ball((window_width / 2), 10, 10, ball_x_vel, ball_y_vel)
bot_bat = Bat(window_height - 10, window_height, window_width / 2 - 40, window_width / 2 + 40)
score = Score(x_pos=(window_width - 150), y_pos=10)
appExit = False


# Application loop
while not appExit:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            appExit = True
            pygame.quit()
            quit()

        # Move the bottom bat left or right if commanded by the human player
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                bot_bat.accelerate("left")
            if event.key == pygame.K_RIGHT:
                bot_bat.accelerate("right")
        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT:
                bot_bat.decelerate("left")
            if event.key == pygame.K_RIGHT:
                bot_bat.decelerate("right")

    bot_bat.move()
    ball.move()
    ball.handle_bounce(bot_bat, score)

    # Update display
    window.fill(BLACK)
    ball.draw()
    bot_bat.draw()
    score.draw()
    pygame.display.update()
    clock.tick(60)


print("TODO")
