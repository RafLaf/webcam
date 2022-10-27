import cv2
import numpy as np

def draw_indicator(frame, percentages, shot_frames,font,scale):
    def percentage_to_color(p):
        return 0,255 - (255 * p), 255 * p
    height, width, _ = frame.shape
    # config
    levels = 50
    level_width = width //10
    level_height = 5
    shift_y = int(height*0.4)
    # draw
    
    #cv2.rectangle(img, (10, img.shape[0] - (indicator_height + 10)), (10 + indicator_width, img.shape[0] - 10), (0, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (20  , shift_y - level_height * (levels+10) ), (20 + level_width*(percentages.shape[0]-1) + level_width -10,shift_y - level_height * (levels+1)  ) ,(0, 0, 0), cv2.FILLED)
    cv2.rectangle(frame, (20  , shift_y + level_height * 1 ), (20 + level_width*(percentages.shape[0]-1) + level_width -10,shift_y + level_height * 10  ) ,(0, 0, 0), cv2.FILLED)
    for k in range(percentages.shape[0]):
        images = shot_frames[k]
        s = images[0].shape
        #frame[20 + level_width*k :20 + level_width*k +s[0] , shift_y + level_height * 10:shift_y + level_height * 10+s[1]] = image
        y_start_img = shift_y 
        x_start_img = 15+level_width*k  
        for n_shot in range(len(images)):
            if y_start_img + s[0] + n_shot*(s[0]+10)< frame.shape[0]:
                frame[ y_start_img + n_shot*(s[0]+10):y_start_img + s[0] + n_shot*(s[0]+10), x_start_img:x_start_img + s[1]] = images[n_shot] #.reshape(s[1], s[0], -1)

        img_level = int(percentages[k] * levels)
        #cv2.putText(frame, str(np.round(percentages[k].item(),2)*100)+'%', (20 + level_width*k  , shift_y - level_height * (levels+3)), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'{np.round(100*percentages[k].item(), 2)}%', (20 + level_width*k  , shift_y - level_height * (levels+3)), font, scale, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.rectangle(frame,(20 + level_width*k , shift_y - levels* level_height), (20 + level_width*k + level_width -10,shift_y  ) , (0,0,0), cv2.FILLED)
        for i in range(img_level):
            level_y_b = shift_y - i * level_height
            start_point = (20 + level_width*k , level_y_b - level_height)
            end_point =  (20 + level_width*k + level_width -10 , level_y_b)
            cv2.rectangle(frame, start_point, end_point , percentage_to_color(i / levels), cv2.FILLED)
            #cv2.rectangle(frame,start_point, end_point, percentage_to_color(i / levels), cv2.FILLED)
            #if i==0:
            #    cv2.putText(frame, str(k), (end_point[0] -level_width//2, end_point[1]+40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
