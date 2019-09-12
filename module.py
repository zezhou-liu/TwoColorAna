import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans

# import the essential packages: numpy as np, matplotlib.pyplot as plt, sklearn.decomposition.PCA as pca
# File container overall is dictionary. The naming format is eccentricity_videoclip_channel. For example: ecc03_2_y1x.
# Please refer bashload function for the naming regulation.
####################### Unit for manual data checking ##########################
class Datahandle:
    def __init__(self, filename):
        # path of data
        self.filename = filename
    def read(self):
        if self.filename[-3:] != 'txt':
            print('File extension is not *.txt')
        else:
            return np.loadtxt(self.filename)
def testplot(x,y):
    #   require import matplotlib.pyplot as plt
    #   require
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if type(x) == str:
        x = Datahandle(x)
    if type(y) == str:
        y = Datahandle(y)
    ax.plot(x.read(), y.read(), '+')

####################### Main process block ####################################
# Bash handle.
class handle:
    def __init__(self):
        return
# Batch unit process
def bashload(main_path):

    # Please arrange your data in the following structure:
    #
    #   -main_path
    #       -2019xxxx_eccxx(the folder format should be time_ecc00)
    #           -video data set(pure number, etc 1,2,3...)
    #               -y1x.txt (y1:yoyo1, y3:yoyo3 etc). TODO: Implement file format modification unit.
    main_path = main_path
    tot_file = {}
    if os.path.exists(main_path):
        os.chdir(main_path)
    else:
        print("No such directory, please check.")
        return ""
    subfolder = os.listdir(main_path)

    for i in subfolder:
        if i[-3:] == 'txt':
            subfolder.remove(i)

    for i in subfolder:
        subpath = main_path+'/'+i
        os.chdir(subpath)
        subsubfolder = os.listdir()
        for j in subsubfolder:
            if i[-3:] == 'txt':
                subsubfolder.remove(i)
        for j in subsubfolder:
            if j[-3:]=="txt":
                continue
            else:
                temp = subpath+'/'+j
                os.chdir(temp)
                prefix = i+'_'+j
                filename = os.listdir()
            for k in filename:
                fname_temp = prefix + '_' + k[:3]
                fname_temp = fname_temp.split('_')[1] + '_' + fname_temp.split('_')[2]+'_' + fname_temp.split('_')[3]
                fpath_temp = temp + '/' + k
                tot_file[fname_temp] = np.loadtxt(fpath_temp)
    a = handle
    a.tot_file = tot_file
    return a, tot_file
def bashvector(handle):
    # Calculate the vector separation for each video
    # tot_file: dictionary of data. Generated by bashload function.
    try:
        tot_file = handle.tot_file
        tot_vector = {}
        temp = ""
        for i in tot_file:
            prefix = i.split('_')[0]+'_'+i.split('_')[1]
            if prefix != temp:
                temp = prefix
                tot_vector[prefix+'_delx'] = tot_file[prefix+'_y3x']-tot_file[prefix+'_y1x']
                tot_vector[prefix+'_dely'] = tot_file[prefix+'_y3y']-tot_file[prefix+'_y1y']
            else:
                continue
        handle.tot_vector = tot_vector
    except:
        print("No tot_file attribute is defined for current input. Please refer to bashload function.")
        return ""
    return handle, tot_vector
def bashoverlay(handle):
    # over lay videos sharing the same cavity shape
    # tot_vector: delta x,y dictionary. Generated by bashvector function.
    try:
        tot_vector = handle.tot_vector
        temp = ""
        tot_vec_overlay = {}
        for i in tot_vector:
            prefix = i.split('_')[0]
            if prefix != temp:
                temp = prefix
                tot_vec_overlay[temp + '_delx'] = np.array([])
                tot_vec_overlay[temp + '_dely'] = np.array([])

        for i in tot_vector:
            prefix = i.split('_')[0]
            # print(i[-1])
            if i[-1] =='x':
                tot_vec_overlay[prefix + '_delx'] = np.append(tot_vec_overlay[prefix + '_delx'], tot_vector.get(i))
            else:
                tot_vec_overlay[prefix + '_dely'] = np.append(tot_vec_overlay[prefix + '_dely'], tot_vector.get(i))
        handle.tot_vec_overlay = tot_vec_overlay
        return handle, tot_vec_overlay
    except:
        print('No tot_vector attribute is defined for current input. Please refer to bashvector function.')
        return
def bashfree(handle, type):
    # Please refer to 'Entropic Segregation of Polymers under Confinement' Pg107, Eq. 5.3.1. Author: Vorgelegt von Elena Minina.
    # 1.PCA alignment of the eclipse(require sklearn.decomposition.PCA module)
    # 2.Define the x-axis separation by coding the data in pca's principle axis
    # 3.Calculate the landscape
    # type------"v" means input is tot_vector from bashvector, "o" means input is totoverlay from bashoverlay. TODO: Automatically distinguish the file type by its name format.
    #################################################################
    try:
        tot_vector = handle.tot_vector
        tot_free ={}
        sep_projection = {}
        temp = ''
        n_components = 2
        pca = PCA(n_components=n_components)
        for i in tot_vector:
            if type == "v":
                prefix = i.split('_')[0] + '_' + i.split('_')[1]
            elif type == "o":
                prefix = i.split('_')[0]
                tot_vector = handle.tot_vec_overlay
            else:
                print("Please insert the type of input")
                break
            if prefix != temp:
                temp = prefix
                xtemp = tot_vector[prefix+'_delx']
                xtemp = xtemp - np.mean(xtemp)
                ytemp = tot_vector[prefix+'_dely']
                ytemp = ytemp - np.mean(ytemp)
                xtrain = np.transpose(np.array([xtemp, ytemp]))
                # x_sep is a 2*n array where row is principle axis, column is the coding.
                x_sep = np.transpose(pca.fit_transform(xtrain))
                sep_projection[temp + '_sep'] = x_sep
                # free energy
                xhist, xbin = np.histogram(x_sep[0, :], bins = 30)
                xhist_short, xbin_short = np.histogram(x_sep[1, :], bins=30)
                index = np.argwhere(xbin>=0)
                index_short = np.argwhere(xbin_short>=0)
                free = -np.log(xhist/len(x_sep[0, :])) + np.log(xhist[index[0, 0]]/len(x_sep[0, :]))
                free_short = -np.log(xhist_short/len(x_sep[1, :])) + np.log(xhist_short[index_short[0, 0]]/len(x_sep[1, :]))
                tot_free[temp + '_F'] = free
                tot_free[temp+'_Fs'] = free_short
                tot_free[temp + '_bins'] = xbin[0:-1]
                tot_free[temp + '_binss'] = xbin_short[0:-1]
        handle.tot_free = tot_free
        handle.sep_projection = sep_projection
        return handle, tot_free, sep_projection
    except:
        print('No tot_vector or tot_vec_overlay attribute is defined for current input. Please refer to bashvector function.')
        return
def bashfreeplt(handle):
    # tot_free--
    # Here the input has to come from bashfree with type'o'. Otherwise there will be too many lines in the graph. Too disturbing.
    try:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        temp = ''
        legend = []
        tot_free = handle.tot_free
        for i in tot_free:
            if len(i.split('_')) == 2:
                prefix = i.split('_')[0]
            else:
                prefix = i.split('_')[0] + '_' + i.split('_')[1]
            if prefix != temp:
                temp = prefix
                legend.append(temp)
                legend.append(temp+'inv')
                ax.plot(tot_free[prefix+'_bins'], tot_free[prefix+'_F'])
                if prefix != 'ecc0':
                    ax.plot(tot_free[prefix + '_binss'], tot_free[prefix + '_Fs'])
        ax.legend(legend)
        return ax
    except:
        print('No tot_free attribute is defined for current input. Please refer to bashfree function.')
def bashpos(handle):
    # mapping the data from negative axis to positive. Assuming the symmetric.
    try:
        tot_vector = handle.tot_vector
        for i in tot_vector:
            if i.split('_')[-1] == 'delx':
                tot_vector[i] = abs(tot_vector[i])
        return tot_vector
    except:
        print('No tot_vector attribute is defined for current input. Please refer to bashvector function.')
def bashshift(handle, n_clusters = 1, n_init = 5, max_iter = 100, tol=0.001):
    ### Using k-means to shift all the data to 0 ###
    ## TODO: Implement the adaptive n_clusters.
    tot_file = handle.tot_file
    temp = ''
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    km = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, tol=tol, precompute_distances='auto',
                verbose=0, random_state=None, algorithm='full')
    for i in tot_file:
        if i.split('_')[0] != 'ecc09':
            continue
        if temp != (i.split('_')[0] + '_' + i.split('_')[1]):
            temp = i.split('_')[0] + '_' + i.split('_')[1]
            y1x = tot_file[temp + '_y1x']
            y1y = tot_file[temp + '_y1y']
            y3x = tot_file[temp + '_y3x']
            y3y = tot_file[temp + '_y3y']
            yoyo1 = np.transpose(np.array([y1x, y1y]))
            yoyo3 = np.transpose(np.array([y3x, y3y]))
            km.fit(yoyo3)
            y1center = km.cluster_centers_
            ax.plot(y1center[:, 0], y1center[:, 1], '+')
    plt.show()
def clean(handle):
## Manual clean all the data outside the ROI. This will pop up a window and let usr select the ROI.
## Data outside ROI will be deleted.
## Require bashvector.
## TODO:Implement here
    return
###############################################
if __name__=="__main__":
    main_path = "D:/McGillResearch/2019Manuscript_Analysis/Analysis/tplasmid"
    handle, tot_file = bashload(main_path)
    bashshift(handle)
# plt.show()