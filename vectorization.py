"""
for training: convert .FOLD files into images, which then convert into vectors

current plan is to use n=17 (16x16 grid), which would mean a d=136 vector
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
n = 17

def fold2matrix(filename,img=False):
    """
    Read a .fold file, make the connection matrix. if img, save the matrix as an image.
    """
    # read the file name as json object
    with open(filename, 'r') as f:
        data = json.load(f)
    # extract the vertices and edges
    vertices = data['vertices_coords']
    edges = data['edges_vertices']
    edges_assignment = data['edges_assignment']
    #convert vertex coords to grid coords. -200 to 200 -> 0 to n
    #potential problem: points that are near the edge might accidentally be placed on the edge
    #TODO: manually add a check for vertices that are closer to the edge than the nearest internal grid point.  move them to the nearest internal grid point, even at the expense of being a bit further.
    for v in vertices:
        v[0] = (v[0]+200)/400 #normalize to 0 to 1
        v[0] = round(v[0]*(n-1)) #scale to 0 to n-1 and round to nearest grid point
        v[1] = (v[1]+200)/400
        v[1] = round(v[1]*(n-1))

    #make an empty nxn numpy array. there are n**2 vertices, so the array will be n**2 x n**2. For a vertex at grid point (i,j), the corresponding vertex or column is at index i*n+j
    connection_matrix = np.zeros((n**2,n**2,3))
    for i in range(len(edges)):
        #get the two vertices that the edge connects
        v1 = vertices[edges[i][0]]
        v2 = vertices[edges[i][1]]
        #get the assignment of the edge
        assignment = edges_assignment[i]
        # mountain folds are a red pixel
        if assignment == 'M':
            connection_matrix[v1[0]*n+v1[1]][v2[0]*n+v2[1]][0] = 255
            connection_matrix[v2[0]*n+v2[1]][v1[0]*n+v1[1]][0] = 255
        # valley folds are a blue pixel
        elif assignment == 'V':
            connection_matrix[v1[0]*n+v1[1]][v2[0]*n+v2[1]][2] = 255
            connection_matrix[v2[0]*n+v2[1]][v1[0]*n+v1[1]][2] = 255
        #borders are a green pixel
        elif assignment == 'B':
            connection_matrix[v1[0]*n+v1[1]][v2[0]*n+v2[1]][1] = 255
            connection_matrix[v2[0]*n+v2[1]][v1[0]*n+v1[1]][1] = 255

    # print(connection_matrix.tolist())

    #turn the nxnx3 connection matrix into a png file
    if img:
        im = Image.fromarray(connection_matrix.astype(np.uint8),'RGB')
        im.save('trainingData/'+filename.split('/')[1]+'.png')

    return connection_matrix    

fold2matrix('trainingData/empty.fold')
fold2matrix('trainingData/225bird_base.fold')
fold2matrix('trainingData/225bird_frog_base.fold')
fold2matrix('trainingData/225blintzed_bird_base.fold')
fold2matrix('trainingData/225dragon.fold')
fold2matrix('trainingData/225deerf.fold')
fold2matrix('trainingData/225deerm.fold')
fold2matrix('trainingData/bp_turtle.fold')


def fold2vector(filename):
    """
    Read a .fold file, make the connection matrix, and flatten it into a vector. For now, M and V are both 1, anything else is 0.
    """
    connection_matrix = fold2matrix(filename)
    vector = np.zeros((int(0.5*(n**2)*(n**2-1)),1))
    for i in range(n**2):
        for j in range(i):
            if connection_matrix[i][j][0] or connection_matrix[i][j][2] :
                # vector = np.vstack((vector,1))
                vector[int(0.5*i*(i-1) + j)][0] = 1
    print(sum(vector))
    # print(vector.T.tolist())
    return vector
fold2vector('trainingData/bp_turtle.fold')

