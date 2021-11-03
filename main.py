# Imports
import pygame
import pygame.freetype
import random


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
    def __init__(self, centre_x_pos, centre_y_pos, width, height):
        self.centre_x_pos = centre_x_pos
        self.centre_y_pos = centre_y_pos
        self.width = width
        self.height = height
        self.top_edge_pos = centre_y_pos - (height / 2)
        self.bot_edge_pos = centre_y_pos + (height / 2)
        self.left_edge_pos = centre_x_pos - (width / 2)
        self.right_edge_pos = centre_x_pos + (width / 2)
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
        self.centre_x_pos += self.x_velocity

        if self.centre_x_pos - (self.width / 2) < 0:
            self.centre_x_pos = self.width / 2

        elif self.centre_x_pos + (self.width / 2) > window_width:
            self.centre_x_pos = window_width - (self.width / 2)

        self.left_edge_pos = self.centre_x_pos - (self.width / 2)
        self.right_edge_pos = self.centre_x_pos + (self.width / 2)

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


class QLearner:
    def __init__(self, alpha, epsilon, gamma, score):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.state = None
        self.prev_state = None
        self.action = "None"
        self.q = Q()  # look-up table of state-action Q values
        self.score_t_minus_2 = score.value  # Assume initially - preceding scores were equal to the current score
        self.score_t_minus_1 = score.value  # Assume initially - preceding scores were equal to the current score

    # Get the agent's environment state - constituting: bat and ball position and ball velocities. Stored as a string.
    def get_state(self, bat, ball):
        state = ""
        state += "bat_x_pos: " + str(bat.centre_x_pos)
        state += " ball_x_pos: " + str(ball.centre_x_pos)
        state += " ball_y_pos: " + str(ball.centre_y_pos)
        state += " ball_x_vel: " + str(ball.x_vel)
        state += " ball_y_vel: " + str(ball.y_vel)
        return state

    def update(self, bat, ball, score):
        # Infer that the reward for entering the previous state is equal to the delta of the game score
        prev_reward = self.score_t_minus_1 - self.score_t_minus_2
        # Update record of scores for next iteration
        self.score_t_minus_2 = self.score_t_minus_1
        self.score_t_minus_1 = score.value  # considered score of previous time step as current action is still pending

        self.state = self.get_state(bat, ball)

        if self.prev_state:

            # Set the Q-value of the previous state-action pair to the shorthand 'Q_sa'. This is equivalent to the
            # notational element Q[s, a] in the update rule below.
            Q_sa = self.q.get(self.prev_state, self.action)

            # Set max_Q which is a shorthand for the maximum Q value for the current state (considering all possible
            # actions). This is equivalent to the notational element max_a'(Q[s', a']) in the update rule below.
            max_Q = self.q.maxQ(self.state)

            # This is where the Q-value update occurs for the previous state. This is equivalent to the slightly more
            # notationally conventional representation:
            # Q[s, a] <- Q[s, a] + alpha * (r + gamma * max_a'(Q[s', a']) - Q[s, a])
            Q_sa = Q_sa + self.alpha * (prev_reward + self.gamma * max_Q - Q_sa)

            # Store the updated Q-value for the previous state
            self.q.set(self.prev_state, self.action, Q_sa)

        # Update the previous state value to the current state ready for the next Q-learning iteration
        self.prev_state = self.state

        # Based on the current state, determine and update the action. This can be thought of in notational terms as:
        # a = f(s, epsilon)
        self.action = self.actionFunction(self.state)

    # actionFunction() method determines the choice of action - using epsilon greedy method
    #
    # By default an action is chosen based upon the most historically successful action(s) from the specified state (as
    # indicated by the recorded Q-values corresponding to the specified state). Exploration is enforced by instead
    # performing a random action at a proportional frequency corresponding to the value of epsilon supplied, i.e.
    # if epsilon is set to 0.1, a random action will be performed one-tenth of the time.
    def actionFunction(self, state):
        possible_actions = ["left", "right", "none"]

        # Perform exploration via random action, proportional to the value of epsilon provided to the agent.
        if random.random() < self.epsilon:
            chosen_action = random.choice(possible_actions)
            return chosen_action

        # Otherwise, perform exploitation, evaluating all the possible moves based upon recorded Q-values, and selecting
        # the action with the highest q value, or randomly selecting among the best options in the case of a tie.
        else:
            evaluated_actions = {}
            for action in possible_actions:
                evaluated_actions[action] = self.q.get(state, action)
            max_q_value = max(evaluated_actions.values())
            best_actions = []
            for action in evaluated_actions:
                if evaluated_actions[action] == max_q_value:
                    best_actions.append(action)
            chosen_action = random.choice(best_actions)
            return chosen_action


# Defines a Q object. A Q object is a persistent record of state-action Q values (i.e. estimated expected utility values
# of state-action pairs). The implementation of the Q object is essentially a dictionary of dictionaries. Method are
# included that facilitate easily understandable ways to read and write data, neatly handling cases where the specified
# state or action do not yet exist in the records.
class Q:
    def __init__(self):
        self.states = {}

    # The get() method will return the Q value of the specified state-action pair, if it exists in the records. If it
    # does not exist, the method will return 0.
    def get(self, state, action):
        return self.states.get(state, {}).get(action, 0)

    # Set the specified Q value for the specified state-action pair
    def set(self, state, action, value):
        if state not in self.states.keys():
            self.states[state] = {}
        self.states[state][action] = value

    # Return the maximum state-action Q value for a given state. I.e. consider all the recorded actions performed for
    # the given state, identify the action with the highest Q value, and return its Q value. If the specified state has
    # not been recorded before, then return 0.
    def maxQ(self, state):
        if state not in self.states.keys():
            return 0
        else:
            best_action_value = max(self.states[state].values())
            return best_action_value


# Initial setup
ball = Ball((window_width / 2), 10, 10, ball_x_vel, ball_y_vel)
bot_bat = Bat(window_width / 2, window_height - 5, 80, 10)
score = Score(x_pos=(window_width - 150), y_pos=10)
q_learner = QLearner(alpha=0.5, epsilon=0.1, gamma=0.9, score=score)
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

    q_learner.update(bot_bat, ball, score)

    if q_learner.action == "left":
        bot_bat.x_velocity = -bat_acceleration
    elif q_learner.action == "right":
        bot_bat.x_velocity = bat_acceleration
    elif q_learner.action == "none":
        bot_bat.x_velocity = 0

    bot_bat.move()
    ball.move()
    ball.handle_bounce(bot_bat, score)

    test_state = q_learner.get_state(bot_bat, ball)

    # Update display
    window.fill(BLACK)
    ball.draw()
    bot_bat.draw()
    score.draw()
    pygame.display.update()
    clock.tick(150)


print("TODO")
