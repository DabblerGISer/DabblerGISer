##this script is used for converting the quanternion params to Rotation matrix, and also compute what is needed for three.js
from read_binary import read_model, write_model
import numpy as np
from argparse import ArgumentParser
import pandas as pd
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import numpy as np
import torch
from plyfile import PlyData, PlyElement

def readasarray(inputpath):
    with open(inputpath, 'rb') as f:
        plydata = PlyData.read(f)
        del f
    sourcevertex = np.array(plydata.elements[0].data)

    return sourcevertex

if __name__ == '__main__':

    parser = ArgumentParser(
        description="Read and write COLMAP binary and text models"
    )
    parser.add_argument("--input_model", help="path to input model folder")
    parser.add_argument(
        "--input_format",
        choices=[".bin", ".txt"],
        help="input model format",
        default="",
    )
    parser.add_argument("--output_model", help="path to output model folder")
    parser.add_argument(
        "--output_format",
        choices=[".bin", ".txt"],
        help="outut model format",
        default=".txt",
    )
    args = parser.parse_args()
    cameras, images, points3D = read_model(
        path=args.input_model, ext=args.input_format
    )
    # write_model(cameras, images, points3D,
    #             path=args.output_model,
    #             ext=args.output_format)
    plydata = readasarray(args.input_model + '/points3D.ply')
    dataframexyz = pd.DataFrame(plydata)
    xyz = dataframexyz[['x', 'y', 'z']]
    center = xyz.mean()
    center_of_model = np.array(center)


    imagesdf = pd.DataFrame(images)
    DFT = imagesdf.T.values
    idx = np.where(DFT[:, 0] == 2)[0][0]
    first_cam = DFT[idx]
    quanternion = first_cam[1]
    T = first_cam[2]
    qw = quanternion[0]
    qx = quanternion[1]
    qy = quanternion[2]
    qz = quanternion[3]

    R11 = 1 - 2 * (qy ** 2 + qz ** 2)
    R12 = 2 * (qx * qy - qw * qz)
    R13 = 2 * (qx * qz + qw * qy)
    R21 = 2 * (qx * qy + qw * qz)
    R22 = 1 - 2 * (qx ** 2 + qz ** 2)
    R23 = 2 * (qy * qz - qw * qx)
    R31 = 2 * (qx * qz - qw * qy)
    R32 = 2 * (qy * qz + qw * qx)
    R33 = 1 - 2 * (qx ** 2 + qy ** 2)
    R=np.array([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]])
    print('R: ', R)
    print('T: ', T)
    trans = np.array([0.0, 0.0, 0.0])
    scale = 1.0
    zfar = 100.0
    znear = 0.01
    #calculate pos of cam from world2camview
    world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
    print('world2view: ', world_view_transform)
    camera_center = world_view_transform.inverse()[3, :3]
    camera_center = camera_center.cpu().numpy()
    print(camera_center)
    # camera_center = np.array(T)
    lookat = center_of_model
    print(lookat)

    world_up = np.zeros([1, 3])
    world_up[0][1] = 1
    # for i in range(len(train_cameras_2_calculate)):

    N = camera_center - lookat
    # N = lookat
    N = N / np.linalg.norm(N)
    U = np.cross(world_up, N)
    U = U / np.linalg.norm(U)
    V = np.cross(N, U)
    V = V / np.linalg.norm(V)

    V = np.array([0, 0, 0]) - V
    # V_prime = V_prime + V

    Vlist = V.tolist()
    vlist2save = []
    for i in range(0, 3):
        vlist2save.append(float(Vlist[0][i]))
    print(vlist2save)
    vlist2save[2] = -vlist2save[2]

    lookatlist = lookat.tolist()

    list2dump = []
    list2dump.append('camera position: ' + str(camera_center.tolist()) + '\n')
    list2dump.append('camera look at: ' + str(lookatlist) + '\n')
    # list2dump.append('camera look at2: ' + str(lookatlist2) + '\n')
    # list2dump.append('camera look at3: ' + str(lookatlist3) + '\n')
    list2dump.append('camera up direction: ' + str(vlist2save) + '\n')

    with open(args.input_model.replace('sparse/0','/output/train/ours_30000') + '/matrix.txt', 'w') as f:
        # f.write('camera position: ' + str(train_cameras_2_calculate[0].camera_center.tolist()))#position
        # f.write('camera look at: ' + str(lookat))
        # f.write('camera up direction: ' + str(Vlist))
        f.writelines(list2dump)

    # projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FoVx,
    #                                              fovY=self.FoVy).transpose(0, 1).cuda()
    # full_proj_transform = (
    #     self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
