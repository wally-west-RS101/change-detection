from osgeo import ogr
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import os
import glob
from tqdm import tqdm
import geopandas as gp
from shapely.geometry import Polygon
import cv2
import rasterio
import numpy as np
def remove_holes(input_path, output_path, min_area=0.0):

    data = gp.read_file(input_path)

    working_data = data.copy()

    holes = working_data[working_data.geom_type == 'MultiPolygon']

    if min_area > 0.0:
        holes['area'] = holes.area
        small_holes = holes[holes['area'] < min_area]
        working_data = working_data[~working_data.isin(small_holes)]
    without_holes = working_data.copy()
    without_holes['geometry'] = without_holes['geometry'].apply(lambda geom: Polygon(geom.exterior))
    without_holes.to_file(output_path)
    return output_path

def raster_to_vecto(input_path,out_path,min_area_delete):

    raster = gdal.Open(input_path)
    band = raster.GetRasterBand(1)
    band.ReadAsArray()
    proj = raster.GetProjection()
    shp_proj = osr.SpatialReference()
    shp_proj.ImportFromWkt(proj)
    output_file = out_path
    call_drive = ogr.GetDriverByName('ESRI Shapefile')
    create_shp = call_drive.CreateDataSource(output_file)
    shp_layer = create_shp.CreateLayer('layername', shp_proj)
    new_field = ogr.FieldDefn(str('ID'), ogr.OFTInteger)
    shp_layer.CreateField(new_field)
    gdal.Polygonize(band, None,shp_layer, 0, [], callback=None)
    create_shp.Destroy()
    raster = None

    img = gp.read_file(output_file)
    old_crs = img.crs
    new_crs = 'EPSG:3857'
    img = img.to_crs(new_crs)
    max = img.geometry.area.max()
    
    if len(img) <= 1:
        os.remove(input_path)
    else:
        
        for index,row in img.iterrows():
            # print(row['geometry'].area)
            if row['geometry'].area == max:
                
                img.drop(index, inplace = True)
            elif row['geometry'].area < min_area_delete:
            
                img.drop(index, inplace = True)
            elif row['ID'] == 0:
                
                img.drop(index, inplace = True)
           
        if len(img) > 0:
            
            img = img.to_crs(old_crs)
            
            img.to_file(out_path)
        else:
            os.remove(out_path)
            return None
    
        return out_path
    
def delete_shadow_with_shp(path_shp_A, path_shp_B, out_path_shp):
    shp_a = gp.read_file(path_shp_A)
    shp_b = gp.read_file(path_shp_B)
    
    sindex_b = shp_b.sindex
    result_A = []

    for idx_a, geometry_a in shp_a.iterrows():
        possible_matches_index = list(sindex_b.intersection(geometry_a['geometry'].bounds))
        possible_matches = shp_b.iloc[possible_matches_index]

        if not possible_matches.empty:
            
            intersecting_geoms = possible_matches[possible_matches.geometry.intersects(geometry_a['geometry'])]

            if not intersecting_geoms.empty:
                result_A.append(geometry_a)
    selected_from_shp_a = gp.GeoDataFrame(result_A, crs=shp_a.crs)
    result = gp.overlay(shp_a, selected_from_shp_a, how='difference')
    out = os.path.join(out_path_shp)
    result.to_file(out)
    return out_path_shp 

def Morphology(image, kernel_size):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    img = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel2)
    
    return img


def delete_cloud_shadow(img):
    img[img != 1] = 0
    return img


def intersec_shp(path_shp_a, path_shp_b,out):
    shp_a = gp.read_file(path_shp_a)
    shp_b = gp.read_file(path_shp_b)
    intersec = gp.overlay(shp_a, shp_b, how='intersection')
    sindex_b = shp_b.sindex
   

    result_A = []
    for idx_a, geometry_a in shp_a.iterrows():
        possible_matches_index = list(sindex_b.intersection(geometry_a['geometry'].bounds))
        possible_matches = shp_b.iloc[possible_matches_index]

        if not possible_matches.empty:
            
            intersecting_geoms = possible_matches[possible_matches.geometry.intersects(geometry_a['geometry'])]

            if not intersecting_geoms.empty:
                result_A.append(geometry_a)
 
    if len(result_A) > 0:
        selected_from_shp_a = gp.GeoDataFrame(result_A, crs=shp_a.crs)
    else:
        print('lennnnnnnnnnnnnnnn = 0')
        selected_from_shp_a = gp.GeoDataFrame(columns=['geometry'])
    sindex_a = shp_a.sindex

    result_B = []
    for idx_b, geometry_b in shp_b.iterrows():
        possible_matches_index_b = list(sindex_a.intersection(geometry_b['geometry'].bounds))
        possible_matches_b = shp_a.iloc[possible_matches_index_b]

        if not possible_matches_b.empty:
            # Kiểm tra đa giác của A có giao với các đa giác của B hay không
            intersecting_geoms_b = possible_matches_b[possible_matches_b.geometry.intersects(geometry_b['geometry'])]

            if not intersecting_geoms_b.empty:
                result_B.append(geometry_b)

    # Tạo GeoDataFrame từ kết quả
    if len(result_B) > 0:
        selected_from_shp_b = gp.GeoDataFrame(result_B, crs=shp_a.crs)
    else:
        print('lennnnnnnnnnnnnnnn = 0')
        selected_from_shp_b = gp.GeoDataFrame(columns=['geometry'])

    # selected_from_shp_b = gp.GeoDataFrame(result_B, crs=shp_a.crs)
    if len(result_A) > 0 and len(result_B) > 0:
        print('union')
        union_result = gp.overlay(selected_from_shp_a, selected_from_shp_b, how='union')
        #dislove 
        print('dissloved')
        dissolved_result = union_result.dissolve()
        #multipart to single part
        print('explode')
        singlepart_gdf = dissolved_result.explode()
        singlepart_gdf.to_file(out)
        return out
    else:
        return 0
def processing_raster(out_path_cf,out_path_shadow,out_path_water,out_path_pro,cfg ):
    with rasterio.open(out_path_cf, 'r') as cf:
            with rasterio.open(out_path_shadow, 'r') as sh:
                with rasterio.open(out_path_water, 'r') as wt:
                    img_cf = cf.read()
                    meta = cf.meta
                    img_shadow = sh.read()
                    img_shadow[img_shadow == 1] = 5
                    img_water = wt.read()
                    img_water[img_water == 1] = 6
        
                    img = np.zeros_like(img_cf)
                    img = img_cf + img_shadow + img_water
                    img = img.transpose(1,2,0)
                    img = delete_cloud_shadow(img)
                    img = Morphology(img, cfg.kenel_size_mopho)
                    print(img.shape)
                    with rasterio.open(out_path_pro,'w',compress='lzw',**meta) as dst:
                        dst.write(img[np.newaxis,:,:])
    return out_path_pro