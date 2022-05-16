import cv2
import os
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import math


class frame():
    def __init__(self,img_path,cal_xl_path):
        self.img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        self.height,self.width = self.img.shape
        self.callibration(cal_xl_path)
        self.pre_process()
        self.fragment_data()
        self.filter_fragment_data()
        self.fragment_data_format()
        self.img_fragments()
    def callibration(self,cal):
        self.cal = pd.read_excel(cal)
        self.c = float(self.cal[self.cal.name == 'cal'].val)
        self.a = float(self.cal[self.cal.name == 'a'].val)
        self.b = float(self.cal[self.cal.name == 'b'].val)
        self.rem = self.cal[self.cal.name.str.contains('remove')].val.to_list()
        # parse rem
        for index, i in enumerate(self.rem):
            i = re.sub('\'|\(|\)','',i)
            i = re.split(',',i)
            i = [int(j) for j in i]
            i = tuple(i)
            self.rem[index] = i
    def pre_process(self):
        # fragment threshold
        ret,self.thresh = cv2.threshold(self.img,60,255,cv2.THRESH_BINARY_INV)
        # find contours
        self.contours, hierarchy = cv2.findContours(self.thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # filter out small contours
        self.contours = [i for i in self.contours if len(i)>100]
        # filter out boundary
        self.contours = [i for i in self.contours if cv2.arcLength(i,0) < 10000]
        # draw contour image
        # new white blank image
        cont = 255 * np.ones(shape=(self.img.shape),dtype= np.uint8)
        self.cont_img = cv2.drawContours(cont, self.contours, -1, (0,255,0), 3)
    def centroid(self,f):
        M = cv2.moments(f)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        return((cx,cy))
    def extract_angle(self,rect,box):
        height,width = rect[1]
        if self.isClose(self.dist(box[0],box[1]),width):
            return(-1 * rect[2])
        elif self.isClose(self.dist(box[0],box[1]),height):
            angle = 90 + rect[2]
            return(angle)
        else:
            return()
    def isClose(self,val1,val2):
        if abs(val1 - val2) < 1:
            return(True)
        else:
            return(False)
    def bounding_box(self,f):
        rect = cv2.minAreaRect(f)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        angle = self.extract_angle(rect,box)
        #return(pd.Series({'bounding_box':box,'angle_deg':angle,'centroid_px':rect[0],'fragment_length_mm':rect[1][0],'fragment_width_mm':rect[1][1],'rect':rect}))
        return(box)
    def bounding_box_length(self,b):
        self.lengths = [abs(np.linalg.norm(b[0] - b[1])) , abs(np.linalg.norm(b[1] - b[2]))]
        return(max(self.lengths) * self.c)
    def bounding_box_width(self,b):
        self.lengths = [abs(np.linalg.norm(b[0] - b[1])) , abs(np.linalg.norm(b[1] - b[2]))]
        return(min(self.lengths)*self.c)
    def bounding_box_orientation(self,b):
        self.lengths = [abs(np.linalg.norm(b[0] - b[1])) , abs(np.linalg.norm(b[1] - b[2]))]
        # check if points 0,1 are length 
        if max(self.lengths) == self.lengths[0]:
            # point 1 is max y
            if b[1][1] > b[0][1]:
                # arctan(x2-x1/y2-y1)
                alpha = np.arctan((b[1][0] - b[0][0]) / (b[1][1] - b[0][1]))
            # point 0 is max y
            else:
                # arctan(x2-x1/y2-y1)
                alpha = np.arctan((b[0][0] - b[1][0]) / (b[0][1] - b[1][1]))
        # points 1,2 are length
        else:
            # point 1 is max y
            if b[1][1] > b[2][1]:
                # arctan(x2-x1/y2-y1)
                alpha = np.arctan((b[1][0] - b[2][0]) / (b[1][1] - b[2][1]))
            # point 0 is max y
            else:
                # arctan(x2-x1/y2-y1)
                alpha = np.arctan((b[2][0] - b[1][0]) / (b[2][1] - b[1][1]))
        return(math.degrees(alpha))
    def bounding_box_orientation_2(self,b):
        if len(b)!=4:
            return()

        # find leading point in x direction
        lead = b[b[:,0] == b[:,0].max()][0]
        #mask lead
        b_pts = b[np.all(b != np.array(lead),axis = 1)]
        #find pt 2
        try:
            self.dist_from_lead = {self.dist(pt,lead):i for i,pt in enumerate(b_pts)}
            self.median_dist = np.median([i for i in self.dist_from_lead])
            self.median_index = self.dist_from_lead[ self.median_dist ]
            pt2 = b_pts[ self.median_index ]
            #arctan(abs(y2-y1)/abs(x2-x1))
            angle = math.degrees(np.arctan(abs(lead[1] - pt2[1]) / abs(lead[0] - pt2[0])))
            # if y1 > y2 angle is neg
            if lead[1] > pt2[1]:
                angle = angle * -1
            # if y2 > y1 angle is pos
            return(angle)
        
        except KeyError:
            return()
    def dist(self,tup1,tup2):
        return(math.sqrt( (tup1[0] - tup2[0])**2 + (tup1[1] - tup2[1])**2 ))
    def dist_to_screw(self,centroid):
        dist_screw = abs(centroid[0] - self.b) #px
        return(dist_screw * self.c)
    def dist_to_2_screws(self,centroid):
        dist_screw = abs(centroid[0] - self.a) #px
        return(dist_screw * self.c)
    def dist_to_remove_points(self,centroid):
        self.remove_distance_list = []
        for pt in self.rem:
            self.remove_distance_list.append(math.sqrt( (centroid[0] - pt[0])**2 + (centroid[1] - pt[1])**2 ))  
        return(min(self.remove_distance_list) * self.c)
    def fragment_data(self):
        self.fragments = pd.DataFrame({'fragments':self.contours})
        self.fragments['contour_length_px'] = self.fragments.fragments.apply(lambda f: cv2.arcLength(f,True))
        self.fragments['area_px'] = self.fragments.fragments.apply(lambda f: cv2.contourArea(f))
        self.fragments['centroid_px'] = self.fragments.fragments.apply(self.centroid)
        self.fragments['bounding_box'] = self.fragments.fragments.apply(self.bounding_box)
        #self.fragments[['bounding_box','angle_deg','centroid_px','fragment_length_mm','fragment_width_mm','rect']] = self.fragments.fragments.apply(self.bounding_box)
        self.fragments['fragment_length_mm'] = self.fragments.bounding_box.apply(self.bounding_box_length)
        self.fragments['fragment_width_mm'] = self.fragments.bounding_box.apply(self.bounding_box_width)
        self.fragments['dist_to_single_screw_mm'] = self.fragments.centroid_px.apply(self.dist_to_screw)
        self.fragments['dist_to_double_screw_mm'] = self.fragments.centroid_px.apply(self.dist_to_2_screws)
        #self.fragments['dist_to_remove_points_mm'] = self.fragments.centroid_px.apply(self.dist_to_remove_points)
    def filter_fragment_data(self):
        # filter out small area
        self.fragments = self.fragments[self.fragments.area_px > 5000]
        #filter out wide fragments
        #self.fragments = self.fragments[self.fragments.fragment_width_mm < 12]
        #filter out remove points
        #self.fragments = self.fragments[self.fragments.dist_to_remove_points_mm != self.fragments.dist_to_remove_points_mm.min()]
        #filter out screws
        #self.fragments = self.fragments[self.fragments.dist_to_single_screw_mm > 1]
        #self.fragments = self.fragments[self.fragments.dist_to_double_screw_mm > 5]
    def fragment_data_format(self):
        #reindex
        self.fragments = self.fragments.reset_index(drop = True)
        #fragment_index
        self.fragments['fragment_index'] = self.fragments.apply(lambda f:int(f.name) + 1 , axis = 1)
        self.fragments['angle_deg'] = self.fragments.bounding_box.apply(self.bounding_box_orientation_2)
        self.frame_data = self.fragments[['fragment_index','dist_to_single_screw_mm','fragment_length_mm','fragment_width_mm','angle_deg']]
    def draw_fragment_contours(self,box):
        cv2.drawContours(self.img,[box],0,(255,255,255),20)
    def write_fragment_number(self,frag):
        font = cv2.FONT_HERSHEY_SIMPLEX
        loc = (int(frag.centroid_px[0]), int(frag.centroid_px[1]) + 500)
        cont = str(frag.fragment_index)
        cv2.putText(self.img,cont,loc,font,10,(255,255,255),10)
    def img_fragments(self):
        self.fragments.bounding_box.apply(self.draw_fragment_contours)
        #self.fragments.apply(self.write_fragment_number,axis = 1)
    def img_save(self,im_dir_path,name):
        cv2.imwrite(f'{im_dir_path}/{name}.jpg',self.cont_img)
    def add_countour(self):
        #define the events for the 
        # mouse_click. 
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', self.img) 
        def sample_neighbor(img, xy_tuple ):
            x,y = xy_tuple
            samples = [img[x+i,y+j] for i in list(range(-10,11,2)) for j in list(range(-10,11,2))]
            return(int(np.average(samples)))
        def fill_area(img, xy_tuple):
            x,y = xy_tuple
            fill = cv2.floodFill(img, None, (x,y), 255)
            return(fill)
        def mouse_click(event, x, y, flags, param): 
            if event == cv2.EVENT_LBUTTONDOWN: 
                print(f'({x},{y}) --> {self.img[x,y]} , {sample_neighbor(self.img,(x,y))}')
                fill = fill_area(self.fill,(x,y))
                disp(fill)
        cv2.setMouseCallback('image', mouse_click) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
class ballistic_test():
    def __init__(self,test,img1_path,img2_path,cal1_path,cal2_path,im_dir_path):
        self.im_dir_path = im_dir_path
        self.test_index = test
        self.f1 = frame(img1_path,cal1_path)
        self.f2 = frame(img2_path,cal2_path)
        self.bt_fragments()
        self.save_images()
    def disp2(self):
        plt.imshow(self.f1.img, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.savefig(f'{im_dir_path}/f1.png')
        plt.show()
        plt.imshow(self.f2.img, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.savefig(f'{im_dir_path}/f2.png')
        plt.show()
    def disp(self,img):
        plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
    def filter_fragment_list(self):
        # used fragments list
        self.f1.used_frag_list = self.test[self.test.frame1.notna()].frame1.to_list()
        self.f2.used_frag_list = self.test[self.test.frame2.notna()].frame2.to_list()
        # mask non used fragments
        self.f1.used_mask = self.f1.fragments.fragment_index.isin(self.f1.used_frag_list)
        self.f2.used_mask = self.f2.fragments.fragment_index.isin(self.f2.used_frag_list)
        # filter list for used fragments
        self.f1.fragments = self.f1.fragments[self.f1.used_mask]
        self.f2.fragments = self.f2.fragments[self.f2.used_mask]
    def renumber_fragment_frames(self):
        # renumber fragment list
        self.frag_list_1 = []
        for index,row in self.f1.fragments.iterrows():
            self.frag_list_1.append(int(self.test[self.test.frame1 == row.fragment_index].frag))
        self.frag_list_2 = []
        for index,row in self.f2.fragments.iterrows():
            self.frag_list_2.append(int(self.test[self.test.frame2 == row.fragment_index].frag))
        self.f1.fragments.fragment_index = self.frag_list_1
        self.f2.fragments.fragment_index = self.frag_list_2
        # write number on picture
        self.f1.fragments.apply(self.f1.write_fragment_number,axis = 1)
        self.f2.fragments.apply(self.f2.write_fragment_number,axis = 1)
    def post_process_fragment_frames(self):
        # sort, reindex
        self.f1.fragments = self.f1.fragments.sort_values(by = ['fragment_index'])
        self.f1.fragments = self.f1.fragments.reset_index(drop = True)
        self.f2.fragments = self.f2.fragments.sort_values(by = ['fragment_index'])
        self.f2.fragments = self.f2.fragments.reset_index(drop = True)
    def bt_df_merge(self):
        cols = ['fragment_index','fragment_length_mm','fragment_width_mm','centroid_px','angle_deg']
        self.fragments = pd.merge(left = self.f1.fragments[cols],
                                  right = self.f2.fragments[cols],
                                  left_on = 'fragment_index', 
                                  right_on = 'fragment_index', 
                                  how = 'outer',
                                  suffixes=('_1', '_2'))
    def fragment_velocity(self,row):
        self.fragments['distance_mm'] = self.fragments.dropna().apply(lambda row: 400 + 
                                                                                self.f1.c * (self.f1.b  + row.centroid_px_2[0]) - 
                                                                                self.f2.c * (self.f2.b  + row.centroid_px_1[0]) , axis = 1)
        self.fragments['velocity_m/s'] = self.fragments.dropna().apply(lambda row: (1000 * (row.distance_mm / self.test.delay.unique()[0])) ,axis = 1) 
    def format(self,series):
        form_list = series.dropna().to_list()
        form = ' , '.join(["{:.2f}".format(i) for i in form_list])
        return(form)
    def format_fragment_index(self,series):
        form_index_list = series.dropna().to_list()
        form_index_list = [i for i in range(1,len(form_index_list)+1)]
        form_index = ' , '.join(["{:d}".format(i) for i in form_index_list])
        return(form_index)
    def save_images(self):
        self.imgs = [self.f1.img, self.f2.img]
        self.req_shape = (min([i.shape[0] for i in self.imgs]) , min([i.shape[1] for i in self.imgs]))
        self.imgs = [cv2.resize(i,self.req_shape) for i in self.imgs]
        self.vis = np.concatenate((self.imgs[0], self.imgs[1]), axis=1)
        cv2.imwrite(f'{self.im_dir_path}\\img_dump\\test_{self.test_index}.png', self.vis)
    def bt_fragments(self):
        self.match_df = pd.read_excel(f'{self.im_dir_path}\\match_fragments.xlsx')
        self.test = self.match_df[self.match_df.test == self.test_index]
        # filter f1 fragments, write fragment number on picture, filter all other fragments
        self.filter_fragment_list()
        # renumber fragments
        self.renumber_fragment_frames()
        # post process
        self.post_process_fragment_frames()
        # merge fragment frames into bt df
        self.bt_df_merge()
        # add dely
        self.fragments['delay'] = float(self.test.delay.unique())
        # calculate velocity
        self.fragments.apply(self.fragment_velocity,axis = 1)
        # format fragment_lengths
        self.fragment_data = pd.DataFrame()
        self.fragment_data['fragment_index_1'] = pd.Series(self.format_fragment_index(self.fragments.fragment_width_mm_1))
        self.fragment_data['fragment_index_2'] = pd.Series(self.format_fragment_index(self.fragments.fragment_width_mm_2))
        self.fragment_data['fragment_velocities_m/s'] = pd.Series(self.format(self.fragments['velocity_m/s']))
        self.fragment_data['fragment_lengths_mm_1'] = pd.Series(self.format(self.fragments.fragment_length_mm_1))
        self.fragment_data['fragment_widths_mm_1'] = pd.Series(self.format(self.fragments.fragment_width_mm_1))
        self.fragment_data['fragment_angles_deg_1'] = pd.Series(self.format(self.fragments.angle_deg_1))
        self.fragment_data['fragment_lengths_mm_2'] = pd.Series(self.format(self.fragments.fragment_length_mm_2))
        self.fragment_data['fragment_widths_mm_2'] = pd.Series(self.format(self.fragments.fragment_width_mm_2))
        self.fragment_data['fragment_angles_deg_2'] = pd.Series(self.format(self.fragments.angle_deg_2))

if __name__ == "__main__":
    def disp(img):
        plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
    def save(img,name):
        cv2.imwrite(f'{im_dir_path}/{name}.jpg',img)
    def callibration(cal_img,x1c,x2c,file_name, rem = None):
        """
            x1c should be centered around two screws [px]
            x2c should be centered around a single screw [px]
            rem = [(x1,y1),(x2,y2)] center of contours that should be removed [px,px]
        """
        img = cv2.cvtColor(cal_img, cv2.COLOR_GRAY2BGR)
        #x1c = 300 
        #x2c = 1620
        rec =  cv2.rectangle(img,(x1c - 50,2000),(x1c + 50,2500),(255,0,0),10)
        rec2 = cv2.rectangle(img,(x2c - 50,2000),(x2c + 50,2500),(0,0,255),10)
        if rem:
            for i in rem:
                rec2 = cv2.rectangle(img,(i[0] - 300,i[1] - 300),(i[0] + 300,i[1] + 300),(0,255,0),10)
        disp(rec2)
        save(rec2,file_name)
        cal = pd.DataFrame({'name':['a','b','cal'],
                            'val':[x1c,x2c,100/abs(x2c - x1c)],
                            'unit':['px','px','mm/px'] })
        if rem:
            for index, value in enumerate(rem):
                cal_series = { 'name': f'remove_{index}',
                                'val': value,
                                'unit':'(px,px)'}
                cal = cal.append(cal_series,ignore_index = True)

        cal.to_excel(f'{im_dir_path}/{file_name}.xlsx',index = False)

    #image directory path
    im_dir_path = r'C:\Users\micha.vardy\shared\lab_test\img_processing'
    #im_dir_path = '/home/michav/shared/lab_test/img_processing'
    im_folder_path = f'{im_dir_path}\\bt1'
    #im_folder_path = f'{im_dir_path}/bt1'
    im_list = [i for i in os.listdir(im_dir_path) if re.search('TIF',i)]
    #image_paths
    img1 = f'{im_dir_path}\\bt1\\Shot_03_01.TIF'
    img2 = f'{im_dir_path}\\bt1\\Shot_03_02.TIF'
    #breaks angle
    #img4 = r"C:\Users\micha.vardy\shared\lab_test\img_processing\bt1\Shot_04_01.TIF"
    #calibration paths
    cal2 = f'{im_dir_path}/d1_global_callibration_2.xlsx'
    cal1 = f'{im_dir_path}/d1_global_callibration_1.xlsx'
    # test ballistic_test
    bt = ballistic_test(3,img1,img2,cal1,cal2,im_dir_path)

    # test frame
    #f1 = frame(img1,cal1)
    #f2 = frame(img2,cal2)
    #disp(f1.img)
    #disp(f2.img)
    #f1.frame_data
    #f4 = frame(img4,cal1)

