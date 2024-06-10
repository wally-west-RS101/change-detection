from osgeo import ogr
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import os
import glob
from tqdm import tqdm
import geopandas as gp
from shapely.geometry import Polygon



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
        
        print('shp can xoa',input_path)
        os.remove(input_path)
        print('da xoa', input_path)

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
            print('warningggggggggggggggggggggg, check ngay ',out_path)
            os.remove(out_path)
            return None
    
        return out_path