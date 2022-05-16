import os
from bt_img_proc import frame, ballistic_test
import pandas as pd
import re
from tqdm import tqdm
import PIL
from PIL import Image
import cv2


def callibration(row):
    #callibration paths
    cal2 = f'{im_dir_path}\\d1_global_callibration_2.xlsx'
    cal1 = f'{im_dir_path}\\d1_global_callibration_1.xlsx'
    cal3 = f'{im_dir_path}\\d3_global_callibration_1.xlsx'
    cal4 = f'{im_dir_path}\\d3_global_callibration_2.xlsx'
    if row.frame == 1 and row.test < 12:
        return(cal1)
    elif row.frame == 2 and row.test <12:
        return(cal2)
    elif row.frame == 1 and row.test >= 12:
        return(cal3)
    elif row.frame == 2 and row.test >= 12:
        return(cal4)
def init_bt_df(im_df):
    # loop over test number
    test_number = list(range(1,int(len(im_df)/2)+1))
    bt_object_list = []
    for i in tqdm(test_number, desc = 'initializing ballistic test dataframe'):
        test = im_df[im_df.test == i]
        bt = ballistic_test(test = i,
                            img1_path = f'{im_dir_path}\\bt1\\{test.iloc[0].img}',
                            img2_path = f'{im_dir_path}\\bt1\\{test.iloc[1].img}',
                            cal1_path = test.iloc[0].cal_path,
                            cal2_path = test.iloc[1].cal_path,
                            im_dir_path = r'C:\Users\micha.vardy\shared\lab_test\img_processing')
        bt_object_list.append(bt)
    return(pd.DataFrame({'test':test_number,'bt':bt_object_list}))


#image directory path
im_dir_path = r'C:\Users\micha.vardy\shared\lab_test\img_processing'
im_folder_path = f'{im_dir_path}\\bt1'

# extract images
im_list = [i for i in os.listdir(im_folder_path) if re.search('TIF',i)]
# format img_df
im_df = pd.DataFrame({'test':[i for i in range(1,int(len(im_list)/2) + 1) for j in range(1,3)],
                      'frame':list(range(1,3))*int(len(im_list)/2),
                      'img':im_list})

#add callibration data
im_df['cal_path'] = im_df.apply(callibration,axis = 1)
# instantiate ballistic test object dataframe
bt_df = init_bt_df(im_df)
# columns
col = ['fragment_index_1', 'fragment_index_2', 'fragment_velocities_m/s',
       'fragment_lengths_mm_1', 'fragment_widths_mm_1',
       'fragment_angles_deg_1', 'fragment_lengths_mm_2',
       'fragment_widths_mm_2', 'fragment_angles_deg_2']
col_final = ['fragment_index','fragment_velocities_m/s','fragment_lengths_mm',
             'fragment_angles_deg','fragment_widths_mm']
bt_df[col] = bt_df.bt.apply(lambda bt:  bt.fragment_data.iloc[0])
# format
bt_df['fragment_index'] = bt_df.apply(lambda bt:f'{bt.fragment_index_1}\n{bt.fragment_index_2}' ,axis = 1)
bt_df['fragment_lengths_mm'] = bt_df.apply(lambda bt:f'{bt.fragment_lengths_mm_1}\n{bt.fragment_lengths_mm_2}' ,axis = 1)
bt_df['fragment_angles_deg'] = bt_df.apply(lambda bt:f'{bt.fragment_angles_deg_1}\n{bt.fragment_angles_deg_2}' ,axis = 1)
bt_df['fragment_widths_mm'] = bt_df.apply(lambda bt:f'{bt.fragment_widths_mm_1}\n{bt.fragment_widths_mm_2}' ,axis = 1)
bt_df = bt_df[col_final]
#writer
writer = pd.ExcelWriter('test.xlsx', engine='xlsxwriter')
bt_df.to_excel(writer, sheet_name='bt1')
# Get the xlsxwriter objects from the dataframe writer object.
workbook  = writer.book
worksheet = writer.sheets['bt1']
# Add a format to use wrap the cell text.
wrap = workbook.add_format({'text_wrap': True})
# Close the Pandas Excel writer and output the Excel file.
writer.save()










