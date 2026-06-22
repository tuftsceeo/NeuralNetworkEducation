import legoeducation as le
import time

def turn_right(dm, degrees):
    dm.movement_turn_for_degrees(degrees=degrees, direction=le.MOVEMENT_TURN_DIRECTION_RIGHT, speed=10)
    time.sleep(0.5)

def turn_left(dm, degrees):
    dm.movement_turn_for_degrees(degrees=degrees, direction=le.MOVEMENT_TURN_DIRECTION_LEFT, speed=10)
    time.sleep(0.5)

def forwards(dm):
    dm.motor_run(direction=le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE, motor=le.MOTOR_RIGHT, speed=10)
    dm.motor_run(direction=le.MOTOR_MOVE_DIRECTION_CLOCKWISE, motor=le.MOTOR_LEFT, speed=10)
    time.sleep(0.5)


dm = le.DoubleMotor()
cL = le.ColorSensor()
cR = le.ColorSensor()

dm.connect(card_serial="1128", card_color=le.LEGO_COLOR_PURPLE)
cL.connect(card_serial="1128", card_color=le.LEGO_COLOR_PURPLE)
cR.connect(card_serial="1131")

while True:
    command = input("Enter command: ")
    if command == "f":
        forwards(dm)
    elif command == "l":
        turn_left(dm, 10)
    elif command == "r":
        turn_right(dm, 10)
    elif command == "q":
        break
    else:
        continue