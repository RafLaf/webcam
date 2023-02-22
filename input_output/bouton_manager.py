#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#La classe boutons renvoie grâce à la méthode change_state, la touche du clavier qui aurait 
#été tappée si au lieu des boutons nous avions utiliser un clavier.
# Lorsque l'on appuie sur le bouton 1 pour capturer une image, change_state renvoie le numéro de
#la classe à laquelle appartient l'image. Lorsqu'on appuie sur le bouton 2, il y a un changement
#de classe et change_state renvoie toujours le numéro de la classe à laquelle appartient l'image.
#Lorsqu'on appuie sur le bouton 3, change_state renvoie "i" pour indiquer qu'il faut passer à l'inférence
#Lorsqu'on appuie sur le bouton 4, change_state renvoie "r" pour indiquer un reset
#Si aucun pouton n'a été pressé, change_state renvoie "NO_KEY_PRESSED"


# In[16]:


import pynq
from pynq import Overlay


# In[17]:


class BoutonsManager:    
    
    def __init__(self, boutons_gpio):
        self.boutons_gpio = boutons_gpio
        self.key_pressed= "1"
        self.last_state = 0
        
    def change_state(self):
        state = self.boutons_gpio.read()
        
        if state != self.last_state :
            if state !=0 :
                if state == 1 :
                    #prendre une image
                    if self.key_pressed=="r":
                        self.key_pressed = "1"
                    print("Une image de la classe"+self.key_pressed+"doit être capturée")
                    self.last_state = state
                    return(self.key_pressed)
                
                if state == 2 :
                    #changer de classe
                    self.key_pressed = str(int(self.key_pressed)+1)
                    print("Passons maintenant à la classe "+ self.key_pressed)
                    self.last_state = state
                    return(self.key_pressed)
                
                if state == 4 :
                    #faire l'inference
                    self.key_pressed="i"
                    self.last_state = state
                    print("Il est temps de faire l'inférence")
                    return(self.key_pressed)
                
                if state == 8 :
                    #reset
                    self.key_pressed="r"
                    print("Les images ont été effacées, veuillez capturez des images de la classe 1")
                    self.last_state = state
                    return(self.key_pressed)
                    
                
            self.last_state = state
        return("NO_KEY_PRESSED")


# In[18]:


import sys
sys.path.append('/home/xilinx')

# Needed to run inference on TCU
import time
import numpy as np
import pynq
from pynq import Overlay
from tcu_pynq.driver import Driver
from tcu_pynq.architecture import pynqz1

overlay = Overlay('/home/xilinx/jupyter_notebooks/l20leche/base_tensil_hdmi.bit')


# In[19]:


btns = BoutonsManager(overlay.btns_gpio)


# In[20]:


while True :
    time.sleep(1)
    print(btns.change_state())


# In[ ]:




