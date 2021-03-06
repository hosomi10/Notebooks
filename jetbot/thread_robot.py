import time
from Adafruit_MotorHAT import Adafruit_MotorHAT
from .thread_motor import Motor


class Robot:
    
    def __init__(self):
        self.i2c_bus = 1
        self.left_motor_channel = 1
        #self.left_motor_alpha = 1.0
        self.right_motor_channel = 2
        #self.right_motor_alpha =1.0
        self.motor_driver = Adafruit_MotorHAT(i2c_bus=self.i2c_bus)
        self.left_motor = Motor(self.motor_driver, channel=self.left_motor_channel)
        self.right_motor = Motor(self.motor_driver, channel=self.right_motor_channel)
        

    #left_motor = traitlets.Instance(Motor)
    #right_motor = traitlets.Instance(Motor)

    # config
    #i2c_bus = traitlets.Integer(default_value=1).tag(config=True)
    #left_motor_channel = traitlets.Integer(default_value=1).tag(config=True)
    #left_motor_alpha = traitlets.Float(default_value=1.0).tag(config=True)
    #right_motor_channel = traitlets.Integer(default_value=2).tag(config=True)
    #right_motor_alpha = traitlets.Float(default_value=1.0).tag(config=True)
    
    #def __init__(self, *args, **kwargs):
        #super(Robot, self).__init__(*args, **kwargs)
        #self.motor_driver = Adafruit_MotorHAT(i2c_bus=self.i2c_bus)
        #self.left_motor = Motor(self.motor_driver, channel=self.left_motor_channel, alpha=self.left_motor_alpha)
        #self.right_motor = Motor(self.motor_driver, channel=self.right_motor_channel, alpha=self.right_motor_alpha)
    
    #function of _write_value in thread_motor.py
    def write_motor(self):
        self.left_motor._write_value(value = self.left_motor.value)
        self.right_motor._write_value(value = self.right_motor.value)

    #need start_motor
    def start(self):
        self.left_motor.motor_start()
        self.right_motor.motor_start()
   
    def set_motors(self, left_speed, right_speed):
        self.left_motor.value = left_speed
        self.right_motor.value = right_speed
        self.write_motor()
        
    def forward(self, speed=1.0, duration=None):
        self.left_motor.value = speed
        self.right_motor.value = speed
        self.write_motor()

    def backward(self, speed=1.0):
        self.left_motor.value = -speed
        self.right_motor.value = -speed
        self.write_motor()

    def left(self, speed=1.0):
        self.left_motor.value = -speed
        self.right_motor.value = speed
        self.write_motor()

    def right(self, speed=1.0):
        self.left_motor.value = speed
        self.right_motor.value = -speed
        self.write_motor()

    def stop(self):
        self.left_motor.value = 0
        self.right_motor.value = 0
        self.write_motor()
