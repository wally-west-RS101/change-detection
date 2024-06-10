
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
import glob,os
from tqdm import *
import pandas as pd

def cal_intersection(df,df1):
    # intersected_data = gpd.GeoDataFrame(columns=df.columns)
    # for i, poly1 in df.iterrows():
    #     for j, poly2 in df1.iterrows():
    #         intersection = poly1['geometry'].intersection(poly2['geometry'])
    #         if not intersection.is_empty and isinstance(intersection, Polygon):
    #             intersected_data = intersected_data.append(poly1, ignore_index=True)
    intersected_data = gpd.overlay(df, df1, how='intersection')
    return intersected_data

def cal_per_recall_f1(df,df1,df2):
    TP = len(df2)
    FN = len(df) - TP
    FP = len(df1) - TP
    TN = 0

    per = TP/(TP + FP)
    recall = TP/(TP + FN)

    f1 = 2*per*recall/(per + recall)

    return per, recall,f1 , TP,FP,FN,TN





# Hiển thị một số thông tin về dữ liệu lấy ra


# print(len(data_building))





if __name__ == "__main__":

    input_dir_shp_qaqc = '/home/skymap/data/Bahrain_change/allqc/BAOCAO_27_2/QC'
    input_dir_ml = '/home/skymap/data/Bahrain_change/allqc/BAOCAO_27_2/All_out/All'

    sel = 'all'

    per = []
    recall = []
    f1 = []
    set_name = []
    TP = []
    FP = []
    FN = []
    TN = []
    total_ml = []
    total_qc = []
    out_dir = '/home/skymap/data/Bahrain_change/allqc/BAOCAO_27_2/'
    list_dir_shp_qaqc = sorted(glob.glob(os.path.join(input_dir_shp_qaqc,'*.shp')))
    list_input_dir_ml = sorted(glob.glob(os.path.join(input_dir_ml,'*.shp')))
    name_exel = 'Report_allsetkointersec.xlsx'
    # print(list_dir_shp_qaqc)
    # print(list_dir_shp_qaqc)
    # print(list_input_dir_ml)

    for fp_shp_qaqc,fp_shp_ml in zip(list_dir_shp_qaqc,list_input_dir_ml):
        print(fp_shp_qaqc)
        print('mmmmmmmm',fp_shp_ml)
       
        data_qaqc = gpd.read_file(fp_shp_qaqc)
        data_ml = gpd.read_file(fp_shp_ml)
        print(data_qaqc)
        total_ml.append(len(data_ml))
        total_qc.append(len(data_qaqc))
        name = os.path.basename(fp_shp_ml).replace('.shp','')
        if sel =='all':
            intersection = cal_intersection(data_qaqc,data_ml)
            per_,recall_,f1_,TP_,FP_,FN_,TN_ = cal_per_recall_f1(data_qaqc,data_ml,intersection)

            per.append(per_)
            recall.append(recall_)
            f1.append(f1_)
            TP.append(TP_)
            FP.append(FP_)
            FN.append(FN_)
            TN.append(TN_)
        elif sel =='building':

            try:
                data_building = data_qaqc.loc[data_qaqc["change_typ"].isin(["New Building", "Building Demolition",'Rooftop Change','Existing Building Extension'])]
                data1_building = data_ml.loc[data_ml["class"].isin(["building_change"])]

            except KeyError:
                try:

                    data_building = data_qaqc.loc[data_qaqc["CHANGE_TYP"].isin(["New Building", "Building Demolition",'Rooftop Change','Existing Building Extension'])]
                    data1_building = data_ml.loc[data_ml["class"].isin(["building_change"])]

                except KeyError:
                    data_building = data_qaqc.loc[data_qaqc["Change_Typ"].isin(["New Building", "Building Demolition",'Rooftop Change','Existing Building Extension'])]
                    data1_building = data_ml.loc[data_ml["class"].isin(["building_change"])]
            
          
            intersection = cal_intersection(data_building,data1_building)
           
            per_,recall_,f1_,TP_,FP_,FN_,TN_= cal_per_recall_f1(data_building,data1_building,intersection)

            per.append(per_)
            recall.append(recall_)
            f1.append(f1_)
            TP.append(TP_)
            FP.append(FP_)
            FN.append(FN_)
            TN.append(TN_)
        elif sel =='vegetation' :

            data_vege = data_qaqc.loc[data_qaqc["change_typ"].isin(['Vegetation Change'])]
            data1_vege = data_ml.loc[data_ml["class"].isin(["vegetation_change"])]

            intersection = cal_intersection(data_vege,data1_vege)
            per_,recall_,f1_,TP_,FP_,FN_,TN_ = cal_per_recall_f1(data_vege,data1_vege,intersection)

            per.append(per_)
            recall.append(recall_)
            f1.append(f1_)
            TP.append(TP_)
            FP.append(FP_)
            FN.append(FN_)
            TN.append(TN_)
        elif sel =='other' :

            data_other = data_qaqc.loc[data_qaqc["change_typ"].isin(['Other Change','Road Change','Water Body Change'])]
            data1_other = data_ml.loc[data_ml["class"].isin(["vegetation_change"])]

            intersection = cal_intersection(data_other,data1_other)
            per_,recall_,f1_,TP_,FP_,FN_,TN_ = cal_per_recall_f1(data_other,data1_other,intersection)

            per.append(per_)
            recall.append(recall_)
            f1.append(f1_)
            TP.append(TP_)
            FP.append(FP_)
            FN.append(FN_)
            TN.append(TN_)

        set_name.append(name)
    data_ = list(zip(set_name,TP,FP,FN,TN,total_ml,total_qc,per,recall,f1))
    data_frame = pd.DataFrame(data_, columns=['SET', 'TP',"FP","FN","TN","TOTAL ML","TOTAL QAQC",'Percision',"Recall","F1 score"])
    out = os.path.join(out_dir,name_exel)
    data_frame.to_excel(out)
        
