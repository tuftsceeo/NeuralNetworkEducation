import legoeducation as le

m = le.DoubleMotor()
c = le.ColorSensor()

m.connect(card_serial="0049")
c.connect(card_serial="0049")

SCALE = 0.15

try:
    while True:
        read = c.sensor.reflection
        speedL = -read * SCALE
        speedR = (100-read) * SCALE
        m.motor_run(speed=speedL, motor=le.MOTOR_LEFT)
        m.motor_run(speed=speedR, motor=le.MOTOR_RIGHT)

except KeyboardInterrupt:
    c.disconnect()
    m.disconnect()

