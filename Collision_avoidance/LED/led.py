import sys
import Jetson.GPIO as GPIO

#Pin number is listed on image

#may be used
#import RPi.GPIO as GPIO
GPIO.setwarnings(False)

led_pin = 31

#mode = GPIO.BOARD, GPIO.BCM, GPIO.CVM, GPIO.TEGRA_SOC
def main(): 

	GPIO.setmode(GPIO.BOARD)
	mode = GPIO.getmode()
	GPIO.setup(led_pin, GPIO.OUT, initial=GPIO.HIGH)
	#GPIO.setup(pin, GPIO.OUT, initial=GPIO.HIGH)

if __name__ == '__main__':

	main()

for i in range(4):

	a = input('key 1 = turn on led, key 2 = turn off led >> ')
	
	#LOW is on and HIGH is off LED
	if int(a) == 1:

		GPIO.output(led_pin, GPIO.LOW)
		print('led on')

	elif int(a) == 2:

		GPIO.output(led_pin, GPIO.HIGH)
		print('led off')

	else:

		GPIO.output(led_pin, GPIO.HIGH)
		print('ignored')

print('finished')

GPIO.cleanup()
