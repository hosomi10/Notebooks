import atexit
from threading import Lock
from Adafruit_MotorHAT import Adafruit_MotorHAT


class Motor: 
    # config
    #alpha = traitlets.Float(default_value=1.0).tag(config=True)
    #beta = traitlets.Float(default_value=0.0).tag(config=True)

    def __init__(self, driver, channel):
        #super(Motor, self).__init__(*args, **kwargs)  # initializes traitlets
        self.alpha = 1.0
        self.beta = 0.0
        self.value = 0.0
        self._driver = driver
        self._motor = self._driver.getMotor(channel)
        self.running = False
        self.write_lock = Lock()
        atexit.register(self._release)
        
        
    #@traitlets.observe('value')
    def _observe_value(self):
        previous_value = self.value
        while self.running :
            if previous_value != self.value :
                self._write_value(self.value)

            previous_val = self.value
        #self._write_value(change['new'])
    
    def motor_start(self):
        print('motor_start at thread_motor.py')
        self.running = True
        #while loop in _observe_value did not work -> _write_value for each function in thread_robot.py
        #self._observe_value()

    def _write_value(self, value):
        """Sets motor value between [-1, 1]"""
        mapped_value = int(255.0 * (self.alpha * value + self.beta))
        speed = min(max(abs(mapped_value), 0), 255)
        self._motor.setSpeed(speed)
        self.write_lock.acquire()
        if mapped_value < 0:
            self._motor.run(Adafruit_MotorHAT.FORWARD)
        else:
            self._motor.run(Adafruit_MotorHAT.BACKWARD)
        self.write_lock.release()

    def _release(self):
        """Stops motor by releasing control"""
        self._motor.run(Adafruit_MotorHAT.RELEASE)
        self.running = False
        print('motor released')
