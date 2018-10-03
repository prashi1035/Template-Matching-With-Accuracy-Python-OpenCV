''' This program will take the two input arguments original image and template image.
It will draw the rectangle to matched template in the original image and score on the top left corner'''

import sys
import os

import cv2
import numpy as np

class template_matching_cls:

    def __init__(self,org_img,tmp_img):
        self.org_img = org_img
        self.tmp_img = tmp_img
    def template_matching(self):
        ''' Finding the templates in the original image'''
        
        self.org_img = org_img
        self.tmp_img = tmp_img
        img_rgb_org = cv2.imread(org_img)
        img_gray = cv2.cvtColor(img_rgb_org, cv2.COLOR_BGR2GRAY)
        template = cv2.imread(tmp_img,0)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.5
        loc = np.where( res >= threshold)
        font = cv2.FONT_HERSHEY_SIMPLEX        
        list_cord = []
        c = 0
        scores = {}
        for pt in zip(*loc[::-1]):
            scores[pt] = res[pt[1]][pt[0]]
            cord_list = [pt[0],pt[1],pt[0] + h,pt[1] + w,scores[pt]]
            list_cord.append(tuple(cord_list))
            c=c+1
        pick = self.non_max_suppression(np.array(list_cord), 0.1)
        test_img = img_rgb_org
        for pt in pick:
            rounded_score = pt[4]*100
            rounded_score_str = "{0:.2f}".format(round(rounded_score,3))
            print(rounded_score_str)
            font = cv2.FONT_HERSHEY_SIMPLEX
            test_img = cv2.rectangle(test_img,(int(pt[0]),int(pt[1])),(int(pt[0]+w),int(pt[1]+h)),(255, 0, 0), 1)
            test_img = cv2.putText(test_img,rounded_score_str, tuple((int(pt[0]),int(pt[1]))), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite("output.jpg",test_img)
        return "done"
        
    def non_max_suppression(self,boxes, overlapThresh):
        '''Gives the highest score co-ordinates of the matched template'''
        if len(boxes) == 0:
            return []
        pick = []
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        ac = boxes[:,4]
        out = self.nms(boxes, overlapThresh)
        return boxes[out]
                
    def nms(self,dets, thresh):
            x1 = dets[:, 0]
            y1 = dets[:, 1]
            x2 = dets[:, 2]
            y2 = dets[:, 3]
            scores = dets[:, 4]
            areas = (x2 - x1 + 1) * (y2 - y1 + 1)
            order = scores.argsort()[::-1]
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
                w = np.maximum(0.0, xx2 - xx1 + 1)
                h = np.maximum(0.0, yy2 - yy1 + 1)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
                inds = np.where(ovr <= thresh)[0]
                order = order[inds + 1]
            return keep
        
        
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('Usage: python template_match_accuracy.py path_to_image_file path_to_template_image\n')
        sys.exit()
    org_img = sys.argv[1]
    tmp_img = sys.argv[2]
    temp_match = template_matching_cls(org_img,tmp_img)
    temp_match.template_matching()
    
