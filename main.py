# Imports
import pygame


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
appExit = False


# Define objects
class Ball:
    def __init__(self, centre_x_pos, centre_y_pos, radius):
        self.centre_x_pos = centre_x_pos
        self.centre_y_pos = centre_y_pos
        self.radius = radius
        self.color = WHITE
        self.x_vel = 5
        self.y_vel = 5

    def move(self):
        self.centre_x_pos = self.centre_x_pos + self.x_vel
        self.centre_y_pos = self.centre_y_pos + self.y_vel

    def handle_bounce(self):
        ball_left_edge = self.centre_x_pos - self.radius
        ball_right_edge = self.centre_x_pos + self.radius
        ball_top_edge = self.centre_y_pos - self.radius
        ball_bottom_edge = self.centre_y_pos + self.radius

        if ball_left_edge <= 0:
            self.x_vel *= -1
            self.centre_x_pos += -ball_left_edge

        elif ball_right_edge >= window_width:
            self.x_vel *= -1
            self.centre_x_pos += ball_right_edge - window_width

        if ball_top_edge <= 0:
            self.y_vel *= -1
            self.centre_y_pos += -ball_top_edge

        elif ball_bottom_edge >= window_height:
            self.y_vel *= -1
            self.centre_y_pos += ball_bottom_edge - window_height

    def draw(self):
        pygame.draw.circle(window, self.color, (self.centre_x_pos, self.centre_y_pos), self.radius)


# Initial setup
ball = Ball((window_width / 2), (window_height / 2), 10)


# Application loop
while not appExit:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            appExit = True
            pygame.quit()
            quit()

    ball.move()
    ball.handle_bounce()

    # Update display
    window.fill(BLACK)
    ball.draw()
    pygame.display.update()
    clock.tick(60)


print("TODO")
