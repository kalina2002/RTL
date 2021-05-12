import gym
import numpy as np
from gym import error, spaces, utils
from gym.utils import seeding
import os
from scipy import misc
import imageio
from subprocess import call
import time
import timeit
import pandas as pd
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import random

class CandyEnv(gym.Env):
    #metadata = {'render.modes': ['human']}
    def __init__(self):
        self.package='com.example.jeupoo'
        self.device='emulator-5554'
        self.apk='candy-debug.apk'
        self.seed()
        #self.action_space = spaces.Box(np.array([-1, 1, 1]),np.array([-200,150, 5]),dtype=np.int32) #x_range y_range duration
        self.action_space = spaces.Discrete(9*9*4) #row * column * 4
        self.resize_scale=0.1
        self.start_time = timeit.default_timer()
        self.coverage_target = 0.6
        self.reset()
        self.startX0= 109 # the position of the left top corner
        self.startY0 = 327
        self.startX1 = 1366 # the position of the right bottom corner
        self.startY1 = 1645 
        self.role = 0
        self.coverage = 0
        print("Candy:init Done")

    def step(self, action):
        if action is not None:
            startX = self.startX0 + 150 * int(action%(9*4))/4 
            startY = self.startY0 + 150 * int(action/(9*4)) 
            direction = action%4
            if direction == 1:
                x_dis = -200
                y_dis = 0
            elif direction == 2:
                x_dis = 200
                y_dis = 0
            elif direction == 3:
                x_dis = 0
                y_dis = 200
            else:
                x_dis = 0
                y_dis = -200
        
            endX = startX + x_dis
            endY = startY + y_dis

        old_coverage = self._get_current_coverage()
        self._exec('adb shell input swipe ' + str(startX)+' '+str(startY)+' '+ str(endX)+' '+str(endY)+' 1000' )
        new_coverage = self._get_current_coverage()
        reward = new_coverage - old_coverage
        self.coverage = new_coverage
         
        done = self.coverage > self.coverage_target
        return self._get_screen(), reward, done

    #def convert_action(self, p_action):
    #    return p_action.dot(np.transpose(np.array([[-200, 0,0], [0,150,0], [0,0,5]])))[0]

    #def convert_paction(self, action):
    #    return action.dot(np.transpose(np.array([[1/-200, 0,0], [0,1/150,0], [0,0,1/5]])))

    def reset(self):
        self._exec('adb -s %s  uninstall %s'%(self.device, self.package))
        self._exec('adb -s %s install /home/adminadmin/Desktop/Lab/script/%s'%(self.device, self.apk))
        self._exec('adb -s %s shell pm grant %s android.permission.WRITE_EXTERNAL_STORAGE'%(self.device, self.package))
      
        self._exec('adb -s %s shell am start -n %s/com.jeupoo.activities.MainActivity'%(self.device, self.package))
        self._exec('adb shell settings put system accelerometer_rotation 0')
        time.sleep(4)
        self._exec('adb shell input tap 1300 2300')
        time.sleep(1)
        self._exec('adb shell input tap 1200 1431')
        time.sleep(1)
        self._exec('adb shell input tap 713 1181')
        time.sleep(1)
        self._exec('adb shell input tap 740 1200')
        time.sleep(6)
        self.coverage = self._get_current_coverage()
        #print('coverage=%f'%self.coverage)
        print("CandyEnv:reset")
        return self._get_screen()

    def render(self, mode='human', close=False):
        print('CandyEnv:render')


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_screen(self):
        self._exec('adb exec-out screencap -p > state.png')
        img = Image.open("state.png")
        return self._image_to_torch(img) 
    
    def _image_to_torch(self, image):
        width, height = image.size
        re_width = int(width * self.resize_scale)
        re_height = int(height * self.resize_scale)
        img_resized = image.resize((re_width, re_height))
        x = np.ascontiguousarray(img_resized, dtype=np.float32) / 255
        #x = x.reshape((1,) + x.shape)
        return x
        #return np.ascontiguousarray(img_resized, dtype=np.float32) / 255


    def _exec(self, command):
        #print(command)
        call(command, shell=True, stdout=None)


    def _get_current_coverage(self):
        self._exec('adb -s %s shell am broadcast -a vt.edu.jacoco.COLLECT_COVERAGE -p %s >> /dev/null 2>&1'%(self.device, self.package))
        self._exec('adb -s %s pull /sdcard/coverage.ec . >> /dev/null 2>&1'%(self.device))
        self._exec('mv coverage.ec coverage.exec')
        generate_report_cmd = 'java -jar /home/adminadmin/Software/gym/gym/examples/apple/gym_apple/jacoco/lib/jacococli.jar report coverage.exec --csv report.csv --classfiles=/home/adminadmin/Software/CasseBonbons/app/build/intermediates/javac/debug/classes  >> /dev/null 2>&1'
        self._exec(generate_report_cmd)
        df = pd.read_csv("report.csv")
        missed, covered = df[['LINE_MISSED', 'LINE_COVERED']].sum()
        #print(f"Complete in {timeit.default_timer() - start_time} seconds")
        return covered * 1.0 / (missed + covered + 1)
