import time
import numpy as np
import matplotlib.pyplot as plt

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock

# Dqn class
from ai import Dqn
brain = Dqn(5, 3, 0.9)

# prevent right click from creating red points
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

last_x = 0
last_y = 0
n_points = 0
length = 0

# action 0: no rotation, action 1: rotate 20 deg., action 2: rotate -20 deg.
action2rotation = [0, 20, -20]
last_reward = 0
scores = []

first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur, largeur))
    goal_x = 20
    goal_y = largeur - 20
    first_update = False

last_distance = 0


class Car(Widget):
    angle = NumericProperty(0) # initialize car angle
    rotation = NumericProperty(0) # initialize last rotation

    velocity_x = NumericProperty(0) # initialize x velocity
    velocity_y = NumericProperty(0) # initialize y velocity
    velocity = ReferenceListProperty(velocity_x, velocity_y) # velocity vector

    sensor1_x = NumericProperty(0) # initialize x-coordinate of sensor 1
    sensor1_y = NumericProperty(0) # initialize y-coordinate of sensor 1
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y) # first sensor vector

    sensor2_x = NumericProperty(0) # initialize x-coordinate of sensor 2
    sensor2_y = NumericProperty(0) # initialize y-coordinate of sensor 2
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y) # second sensor vector

    sensor3_x = NumericProperty(0) # initialize x-coordinate of sensor 3
    sensor3_y = NumericProperty(0) # initialize y-coordinate of sensor 3
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y) # third sensor vector

    signal1 = NumericProperty(0) # initialize signal received by sensor 1
    signal2 = NumericProperty(0) # initialize signal received by sensor 2
    signal3 = NumericProperty(0) # initialize signal received by sensor 3

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos # update position
        self.rotation = rotation # update rotation
        self.angle = self.angle + self.rotation # update the angle

        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos # update sensor 1 position
        self.sensor2 = Vector(30, 0).rotate((self.angle + 30) % 360) + self.pos # update sensor 2 position
        self.sensor3 = Vector(30, 0).rotate((self.angle - 30) % 360) + self.pos # update sensor 3 position

        # get density of sand around each sensor
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10])) / 400.0
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10])) / 400.0
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10])) / 400.0

        # if sensor 1 is out of the map
        if self.sensor1_x > longueur-10 or self.sensor1_x < 10 or self.sensor1_y > largeur-10 or self.sensor1_y < 10:
            self.signal1 = 1. # sensor 1 detects all sand

        # if sensor 2 is out of the map
        if self.sensor2_x > longueur-10 or self.sensor2_x < 10 or self.sensor2_y > largeur-10 or self.sensor2_y < 10:
            self.signal2 = 1. # sensor 2 detects all sand

        # if sensor 3 is out of the map
        if self.sensor3_x > longueur-10 or self.sensor3_x < 10 or self.sensor3_y > largeur-10 or self.sensor3_y < 10:
            self.signal3 = 1. # sensor 3 detects all sand


class Ball1(Widget):
    pass


class Ball2(Widget):
    pass


class Ball3(Widget):
    pass


class Game(Widget):
    car = ObjectProperty(None) # car object
    ball1 = ObjectProperty(None) # sensor 1 object
    ball2 = ObjectProperty(None) # sensor 2 object
    ball3 = ObjectProperty(None) # sensor 3 object

    def serve_car(self):
        self.car.center = self.center # start car at center of map
        self.car.velocity = Vector(3, 0) # car starts with horizontal velocity

    def update(self, dt):
        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur

        longueur = self.width # width of map
        largeur = self.height # height of map
        if first_update:
            init()

        xx = goal_x - self.car.x # delta in x
        yy = goal_y - self.car.y # delta in y
        orientation = Vector(*self.car.velocity).angle((xx, yy)) / 180.0 # direction of car w.r.t goal
        last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        action = brain.update(last_reward, last_signal) # playing the action from our AI
        scores.append(brain.score()) # store score of action
        rotation = action2rotation[action] # convert action played (0,1,2) into rotation angle
        self.car.move(rotation)

        # calculate distance to goal
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1 # update sensor 1 position
        self.ball2.pos = self.car.sensor2 # update sensor 2 position
        self.ball3.pos = self.car.sensor3 # update sensor 3 position

        # penalty for driving in sand
        if sand[int(self.car.x), int(self.car.y)] > 0: # car is on the sand
            self.car.velocity = Vector(1, 0).rotate(self.car.angle) # car is slowed down
            last_reward = -1
        else:
            # living penalty
            self.car.velocity = Vector(3, 0).rotate(self.car.angle)
            last_reward = -0.2
            # reward for going to goal
            if distance < last_distance:
                last_reward = 0.1

        # penalties for being on screen edges
        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1

        if self.car.x > self.width-10:
            self.car.x = self.width-10
            last_reward = -1

        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1

        if self.car.y > self.height-10:
            self.car.y = self.height-10
            last_reward = -1

        # update last distance from car to goal
        if distance < 100:
            goal_x = self.width - goal_x
            goal_y = self.height - goal_y

        last_distance = distance


class MyPaintWidget(Widget):
    # draw sand with left mouse click
    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8, 0.7, 0)
            d = 10.0
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x), int(touch.y)] = 1

    # draw sand when dragging left click mouse
    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1
            density = n_points / (length)
            touch.ud['line'].width = int(20*density + 1)
            sand[int(touch.x)-10:int(touch.x)+10, int(touch.y)-10:int(touch.y)+10] = 1
            last_x = x
            last_y = y


class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60)
        self.painter = MyPaintWidget()
        # make buttons with callbacks
        clearbtn = Button(text='clear')
        savebtn = Button(text='save',pos=(parent.width,0))
        loadbtn = Button(text='load',pos=(2*parent.width,0))
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        # add buttons to screen
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur, largeur))

    def save(self, obj):
        print('saving brain...')
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print('loading last saved brain...')
        brain.load()


if __name__ == '__main__':
    CarApp().run()
