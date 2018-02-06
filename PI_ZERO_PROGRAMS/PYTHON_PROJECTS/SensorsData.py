# Module to deploy on Pi - check data from MULTIPLE sensors
# In my project I used 5 ultrasound sensors
# Import libraries

import time
import pigpio

# Explicitly inform about the speed of sound 
SOS= 340.29

# Class Sonar - easily create sensors

class Sonar:

   # Define sensor - trigger, echo and pi object of pigpio library
   def __init__(self, pi, trigger, echo):
      self.pi = pi
      self.trig = trigger

      self._distance = 999.9
      self._one_tick = None

      if trigger is not None:
         pi.set_mode(trigger, pigpio.OUTPUT)

      pi.set_mode(echo, pigpio.INPUT)

      self._cb = pi.callback(echo, pigpio.EITHER_EDGE, self._cbf)

   # Callback function - to take measures of sensor
   def _cbf(self, gpio, level, tick):
      if level == 1:
         self._one_tick = tick
      else:
         if self._one_tick is not None:
            ping_micros = pigpio.tickDiff(self._one_tick, tick)
            self._distance = (ping_micros * SOS) / 2e4
            self._one_tick = None

   def trigger(self):
      self._distance = 999.9
      self._one_tick = None

      if self.trig is not None:
         self.pi.gpio_trigger(self.trig, 15) # 15 micros trigger pulse

   def read(self):
      return self._distance

   def cancel(self):
      self._cb.cancel()

# Class SonarArray - to easily organize sensors and get measures
class SonarArray():

    def __init__(self):
        self.sonars = []
        self._distances = []
    
    def dist_return(self):
        return self._distances

    def add_sonar(self, sonar: Sonar):
        self.sonars.append(sonar)
    
    def delete_sonar(self, index):
        self.sonars.pop(index)

    def get_distances(self):
        for s in self.sonars:
            s.trigger()

        time.sleep(0.03)
        
        for s in self.sonars:
            self._distances.append(s.read())
        
        dist_return

#Main part - to check if everything is OK
if __name__ == "__main__":

    import time
    import pigpio

    pi = pigpio.pi()

    if not pi.connected:
        exit()

    S = []
    S.append(Sonar(pi, None, 5))
    S.append(Sonar(pi, None, 6))
    S.append(Sonar(pi, None, 13))
    S.append(Sonar(pi, None, 19))
    S.append(Sonar(pi, 12, 26))

    end =  time.time() + 100.0

    r = 1

    try:
        while time.time() < end:
            for s in S:
                s.trigger()

            time.sleep(0.03)

            for s in S:
                print("{} {:.1f}".format(r, s.read()))
            
            time.sleep(0.2)

            r += 1
            print("=====================================")

    except KeyboardInterrupt:
        pass



    print("\ntidying up")

    for s in S:
        s.cancel()

    pi.stop()
