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


# Application loop
while not appExit:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            appExit = True
            pygame.quit()
            quit()

    window.fill(BLACK)
    pygame.display.update()
    clock.tick(60)


print("TODO")
