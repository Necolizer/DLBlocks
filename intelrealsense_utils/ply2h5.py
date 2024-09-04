import h5py
import numpy as np
from plyfile import PlyData, PlyElement
import open3d as o3d
import pandas as pd


def ply2h5(ply_path, h5_path):
    '''
    将ply文件转换为h5文件
    ply文件中的点云xyz坐标是vertex 这里只转了这一部分到h5文件
    '''
    f = h5py.File(h5_path, 'w')

    plydata = PlyData.read(ply_path)
    print(len(plydata['vertex'].data)) # 点云的数量

    a_data = np.zeros((len(plydata['vertex'].data), 3))

    for j in range(len(plydata['vertex'].data)):
        a_data[j] = [plydata['vertex']['x'][j], plydata['vertex']['y'][j], plydata['vertex']['z'][j]]

    data = f.create_dataset("data", data = a_data)

def h52ply(h5_path, ply_path):
    '''
    将h5文件转换为ply文件
    h5文件中的点云xyz坐标是data 这里只转了这一部分到ply文件
    '''
    f = h5py.File(h5_path, 'r')

    print(f['data'].shape)

    t = f['data'].value

    vertex = np.empty(len(t), dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    for i in range(len(t)):
        vertex[i]['x'] = t[i][0]
        vertex[i]['y'] = t[i][1]
        vertex[i]['z'] = t[i][2]
    el = PlyElement.describe(vertex, 'vertex')

    PlyData([el], text=True).write(ply_path)

def read_ply_pointcloud(ply_path):
    plydata = PlyData.read(ply_path)

    df = pd.DataFrame(plydata['vertex'].data)

    # print(df.shape)

    # data_np = np.zeros(df.shape, dtype=np.float)  # 初始化储存数据的array
    # property_names = plydata['vertex'].data[0].dtype.names  # 读取property的名字
    # for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型
    #     data_np[:, i] = df[name]

    return df

def ply2pcd(ply_path, pcd_path):
    f = o3d.io.read_point_cloud(ply_path)
    o3d.io.write_point_cloud(pcd_path, f)

def pcd2ply(pcd_path, ply_path):
    f = o3d.io.read_point_cloud(pcd_path)
    o3d.io.write_point_cloud(ply_path, f)

def npy2ply(npy_path, ply_path):
    point_numpy = np.load(npy_path)
    print(point_numpy.shape)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(point_numpy)

    o3d.io.write_point_cloud(ply_path, cloud)

def ply2npy(ply_path, npy_path):
    cloud = o3d.io.read_point_cloud(ply_path)
    point_numpy = np.asarray(cloud.points)

    print(point_numpy.shape)

    np.save(npy_path, point_numpy)


if __name__ == '__main__':
    # ply2h5(r'D:\PersonalProject\PointCloud2023\linx\1-G17040003-PointCloud.ply', r'D:\PersonalProject\PointCloud2023\00.h5')
    # h52ply(r'D:\PersonalProject\PointCloud2023\00.h5', r'D:\PersonalProject\PointCloud2023\00.ply')

    # df1 = read_ply_pointcloud(r'D:\PersonalProject\PointCloud2023\00.ply') # 转换后
    # df2 = read_ply_pointcloud(r'D:\PersonalProject\PointCloud2023\linx\1-G17040003-PointCloud.ply') # 转换前

    # print(df1.equals(df2)) # True

    npy2ply(r'D:\PersonalProject\2023-PointCloud\wyh\npy\04554684-fcc0bdba1a95be2546cde67a6a1ea328.npy', 
    r'D:\PersonalProject\2023-PointCloud\wyh\npy\04554684-fcc0bdba1a95be2546cde67a6a1ea328.ply')