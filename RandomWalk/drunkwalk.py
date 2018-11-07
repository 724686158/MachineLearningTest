# -*- encode=utf8 -*-

import random
import turtle
import math

OX = 0
OY = 0
RAD = 200
STEP = 30

def drawCircle (drunkard):
    drunkard.speed(100)
    drunkard.hideturtle()
    drunkard.pensize(4)
    drunkard.color("black")
    drunkard.penup()
    drunkard.setpos(OX, OY - RAD)
    drunkard.pendown()
    drunkard.circle(RAD, 330)
    drunkard.up()
    drunkard.setpos(OX, OY)
    drunkard.color("green")
    drunkard.dot(10)
    drunkard.down()

def drunkWalk (drunkard):
    drunkard.pensize(2)
    drunkard.showturtle()
    drunkard.speed(7)
    con = 0
    while True:
        con = con + 1
        xlim = (drunkard.xcor())
        ylim = (drunkard.ycor())
        angle = random.uniform(0, 360)
        dist = int(random.uniform(STEP, STEP * 1.5))
        drunkard.setheading(angle)
        drunkard.forward(dist)
        xbound = (drunkard.xcor())
        ybound = (drunkard.ycor())
        rad = math.sqrt(xbound**2 + ybound**2)
        if rad <= RAD:
            drunkard.color("green")
            drunkard.goto(xbound, ybound)
        elif (rad > RAD) and (xbound < 0) and (ybound < -RAD * math.sqrt(3) / 2):
            drunkard.color("green")
            drunkard.goto(xbound, ybound)
            drunkard.color("yellow")
            drunkard.dot(10)
            break
        else:
            drunkard.color('red')
            drunkard.goto(xbound, ybound)
            drunkard.speed(1)
            drunkard.goto(xlim, ylim)
            drunkard.speed(7)
            drunkard.color("green")
    drunkard.up()
    drunkard.setpos(OX, OY - RAD - 30)
    drunkard.down()
    drunkard.color("black")
    drunkard.hideturtle()
    drunkard.write("醉汉一共走了{}步".format(con))


if __name__ == '__main__':
    atlas = turtle.Turtle()
    drawCircle(atlas)
    drunkWalk(atlas)
    screen = atlas.getscreen()
    screen.exitonclick()
