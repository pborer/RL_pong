# Imports
import sys
import pygame
import pygame.freetype
import random


# Store command-line arguments
command_line_args = sys.argv
# Capture the agent type from the 1st command-line argument supplied, otherwise default to periodic observer type
agent_type = sys.argv[1] if len(sys.argv) > 1 else "periodic_observer"
if agent_type not in ["periodic_observer", "continual_observer"]:
    agent_type = "periodic_observer"


# Define constants
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# Pygame setup
pygame.init()
window_width = 240
window_height = 200
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption('Pong via reinforcement learning')
clock = pygame.time.Clock()
GAME_FONT = pygame.freetype.SysFont("sans-serif", 20)


# Config
bat_acceleration = 5
ball_x_vel = 5
ball_y_vel = 5


# Define objects
class Ball:
    def __init__(self, centre_x_pos, centre_y_pos, radius, ball_x_vel, ball_y_vel):
        self.centre_x_pos = centre_x_pos
        self.centre_y_pos = centre_y_pos
        self.radius = radius
        self.color = WHITE
        self.x_vel = ball_x_vel
        self.y_vel = ball_y_vel
        self.is_at_vertical_interaction = True

    def move(self):
        self.is_at_vertical_interaction = False
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
            self.is_at_vertical_interaction = True

        # Reset ball at top if it falls off the bottom of the screen, decrease score
        elif bot_edge_pos > window_height:
            self.centre_y_pos = self.radius
            self.is_at_vertical_interaction = True
            score.modify(-50)

        # Handle bounce off of bottom bat, increase score
        if (
            bot_edge_pos > bot_bat.top_edge_pos
            and left_edge_pos < bot_bat.right_edge_pos
            and right_edge_pos > bot_bat.left_edge_pos
        ):
            self.y_vel *= -1
            self.centre_y_pos -= bot_edge_pos - bot_bat.top_edge_pos
            self.is_at_vertical_interaction = True
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
    def __init__(self, agent_type, alpha, epsilon, gamma, score):
        self.agent_type = agent_type
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.state = None
        self.prev_state = None
        self.action = "None"
        self.q = Q()  # look-up table of state-action Q values
        self.score_t_minus_2 = score.value  # Assume initially that preceding scores were equal to the current score
        self.score_t_minus_1 = score.value  # Assume initially that preceding scores were equal to the current score

    # Get the agent's environment state - constituting: bat and ball position and ball velocities. Stored as a string.
    def get_state(self, bat, ball):
        state = ""
        state += "bat_x_pos: " + str(bat.centre_x_pos)
        state += " ball_x_pos: " + str(ball.centre_x_pos)
        state += " ball_y_pos: " + str(ball.centre_y_pos)
        state += " ball_x_vel: " + str(ball.x_vel)
        state += " ball_y_vel: " + str(ball.y_vel)
        return state

    def update(self, bat, ball, score, is_greedy=False):
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
        self.action = self.actionFunction(self.state, is_greedy)

    # actionFunction() method determines the choice of action - using epsilon greedy method
    #
    # By default an action is chosen based upon the most historically successful action(s) from the specified state (as
    # indicated by the recorded Q-values corresponding to the specified state). Exploration is enforced by instead
    # performing a random action at a proportional frequency corresponding to the value of epsilon supplied, i.e.
    # if epsilon is set to 0.1, a random action will be performed one-tenth of the time.
    def actionFunction(self, state, is_greedy):

        if self.agent_type == "periodic_observer":
            possible_actions = []
            for direction in ["left", "right"]:
                for duration in range(0, 40):  #TODO hardocded to 40 for now - but should update to dynamically adjust
                    possible_actions.append(direction + ":" + str(duration))

        elif self.agent_type == "continual_observer":
            possible_actions = ["left", "right", "none"]

        # Perform exploration via random action, proportional to the value of epsilon provided to the agent.
        if random.random() < self.epsilon and not is_greedy:
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
q_learner = QLearner(agent_type=agent_type, alpha=0.5, epsilon=0.1, gamma=0.99, score=score)
appExit = False
clock_speed_observation = 60  # Meaning apply updates no faster than 60 frames per second
clock_speed_training = 0  # Meaning apply updates as fast as possible
training_iterations = 0
movement_direction = None
remaining_movement_iterations = 0
processing_mode = "observation"

# Application loop
while not appExit:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            appExit = True
            pygame.quit()
            quit()

        # If enter is pressed, perform a batch of max speed training iterations
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                processing_mode = "training"
            if event.key == pygame.K_BACKSPACE:
                processing_mode = "observation"

    if q_learner.agent_type == "periodic_observer":
        if ball.is_at_vertical_interaction:
            is_greedy = processing_mode == "observation"  # When in observation mode actions are chosen greedily
            q_learner.update(bot_bat, ball, score, is_greedy)
            movement_direction = q_learner.action.split(":")[0]
            remaining_movement_iterations = int(q_learner.action.split(":")[1])

        if movement_direction == "left" and remaining_movement_iterations > 0:
            bot_bat.x_velocity = -bat_acceleration
            remaining_movement_iterations -= 1
        elif movement_direction == "right" and remaining_movement_iterations > 0:
            bot_bat.x_velocity = bat_acceleration
            remaining_movement_iterations -= 1
        else:
            bot_bat.x_velocity = 0

    elif q_learner.agent_type == "continual_observer":
        is_greedy = processing_mode == "observation"  # When in observation mode actions are chosen greedily
        q_learner.update(bot_bat, ball, score, is_greedy)

        if q_learner.action == "left":
            bot_bat.x_velocity = -bat_acceleration
        elif q_learner.action == "right":
            bot_bat.x_velocity = bat_acceleration
        elif q_learner.action == "none":
            bot_bat.x_velocity = 0

    bot_bat.move()
    ball.move()
    ball.handle_bounce(bot_bat, score)

    # Update display
    window.fill(BLACK)
    ball.draw()
    bot_bat.draw()
    score.draw()
    pygame.display.update()

    # Handle variable running speed
    if processing_mode == "training":
        clock.tick(clock_speed_training)
    elif processing_mode == "observation":
        clock.tick(clock_speed_observation)
