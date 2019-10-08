import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.integrate import quad
import json
from scipy.constants import Boltzmann, Avogadro, elementary_charge, epsilon_0
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
    import matplotlib.pyplot as plt
    #   require
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    if type(x) == str:
        x = Datahandle(x)
    if type(y) == str:
        y = Datahandle(y)
    ax.plot(x.read(), y.read(), '+')
def swapper(x, ind):
    ## change the sign of the input data[ind]
    ## input:   x, *.txt file needs to be swapped
    ##          ind: 2 elements array-like data. [l_ind, h_ind]
    filename = x
    path = x.replace(x.split('/')[-1], '')
    os.chdir(path)
    data = np.loadtxt(filename)
    index = np.arange(ind[0], ind[1] + 1)
    data[index] = - data[index]
    np.savetxt(x.split('/')[-1], data)
    print(filename + 'index['+str(ind[0])+','+str(ind[1])+'] has been swapped!')
    return
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
    subfolder = os.listdir()

    for i in subfolder:
        if 'ecc' not in i:
            continue
        subpath = main_path+'/'+i
        os.chdir(subpath)
        subsubfolder = os.listdir()
        for j in subsubfolder:
            if j[-3:] == 'txt' or j[-4:] == 'json':
                subsubfolder.remove(j)
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
                print('Reading files from:', fpath_temp)
                tot_file[fname_temp] = np.loadtxt(fpath_temp)
    a = handle
    a.tot_file = tot_file
    a.main_path = main_path
    return a, tot_file
def bashvector(handle, mode='raw'):
    # Calculate the vector separation for each video
    # tot_file: dictionary of data. Generated by bashload function.
    try:
        if mode == 'raw':
            tot_file = handle.tot_file
        elif mode == 'clean':
            tot_file = handle.tot_file_shift
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
        if mode == 'raw':
            handle.tot_vector = tot_vector
        elif mode == 'clean':
            handle.tot_vector_clean = tot_vector
    except:
        print("No tot_file attribute is defined for current input. Please refer to bashload function.")
        return ""
    return handle, tot_vector
def bashoverlay(handle, mode = 'raw', set = 'vector'):
    # over lay videos sharing the same cavity shape
    # tot_vector: delta x,y dictionary. Generated by bashvector function.
    try:
        if set == 'vector':
            if mode == 'clean':
                tot_vector = handle.tot_vector_clean
            elif mode == 'raw':
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
            if mode == 'raw':
                handle.tot_vec_overlay = tot_vec_overlay
            elif mode == 'clean':
                handle.tot_vec_overlay_clean = tot_vec_overlay

        elif set == 'position':
            if mode == 'clean':
                tot_pos = handle.tot_file_shift
            elif mode == 'raw':
                tot_pos = handle.tot_file
            temp = ""
            tot_vec_overlay = {}
            for i in tot_pos:
                prefix = i.split('_')[0]
                if prefix != temp:
                    temp = prefix
                    tot_vec_overlay[temp + '_y1x'] = np.array([])
                    tot_vec_overlay[temp + '_y1y'] = np.array([])
                    tot_vec_overlay[temp + '_y3x'] = np.array([])
                    tot_vec_overlay[temp + '_y3y'] = np.array([])

            for i in tot_pos:
                prefix = i.split('_')[0]
                # print(i[-1])
                if i[-3:] == 'y1x':
                    tot_vec_overlay[prefix + '_y1x'] = np.append(tot_vec_overlay[prefix + '_y1x'], tot_pos.get(i))
                elif i[-3:] == 'y1y':
                    tot_vec_overlay[prefix + '_y1y'] = np.append(tot_vec_overlay[prefix + '_y1y'], tot_pos.get(i))
                elif i[-3:] == 'y3x':
                    tot_vec_overlay[prefix + '_y3x'] = np.append(tot_vec_overlay[prefix + '_y3x'], tot_pos.get(i))
                elif i[-3:] == 'y3y':
                    tot_vec_overlay[prefix + '_y3y'] = np.append(tot_vec_overlay[prefix + '_y3y'], tot_pos.get(i))
                else:
                    print('Bug in bashvector with set=position')
                    print(i[-3:])
            if (mode == 'raw') and (set == 'vector'):
                handle.tot_vec_overlay = tot_vec_overlay
            elif (mode == 'clean') and (set == 'vector'):
                handle.tot_vec_overlay_clean = tot_vec_overlay
            elif (mode == 'raw') and (set == 'position'):
                handle.tot_pos_overlay = tot_vec_overlay
            elif (mode == 'clean') and (set == 'position'):
                handle.tot_pos_overlay_shift = tot_vec_overlay
        return handle, tot_vec_overlay
    except:
        print('No tot_vector attribute is defined for current input. Please refer to bashvector function.')
    return
def bashfree(handle, type='o'):
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
        return handle, tot_free
    except:
        print('No tot_vector or tot_vec_overlay attribute is defined for current input. Please refer to bashvector function.')
        return
# def bashfreeplt(handle):
#     # tot_free--
#     # Here the input has to come from bashfree with type'o'. Otherwise there will be too many lines in the graph. Too disturbing.
#     try:
#         fig = plt.figure()
#         ax = fig.add_subplot(1,1,1)
#         temp = ''
#         legend = []
#         tot_free = handle.tot_free
#         for i in tot_free:
#             if len(i.split('_')) == 2:
#                 prefix = i.split('_')[0]
#             else:
#                 prefix = i.split('_')[0] + '_' + i.split('_')[1]
#             if prefix != temp:
#                 temp = prefix
#                 legend.append(temp)
#                 legend.append(temp+'inv')
#                 ax.plot(tot_free[prefix+'_bins'], tot_free[prefix+'_F'])
#                 if prefix != 'ecc0':
#                     ax.plot(tot_free[prefix + '_binss'], tot_free[prefix + '_Fs'])
#         ax.legend(legend)
#         return ax
#     except:
#         print('No tot_free attribute is defined for current input. Please refer to bashfree function.')

# def bashpos(handle):
#     # mapping the data from negative axis to positive. Assuming the symmetric.
#     try:
#         tot_vector = handle.tot_vector
#         for i in tot_vector:
#             if i.split('_')[-1] == 'delx':
#                 tot_vector[i] = abs(tot_vector[i])
#         return tot_vector
#     except:
#         print('No tot_vector attribute is defined for current input. Please refer to bashvector function.')
def bashshift(handle, n_init = 5, max_iter = 100, tol=0.001):
    ### Using k-means to shift all the data to 0 ###
    ## TODO: Implement the adaptive n_clusters.
    ## Output: handle.kmeans['ecc0_1']: Kmeans class. Please refer: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    ## for more info.
    tot_file_clean = handle.tot_file_clean
    temp = ''
    kmeans = {}
    for i in tot_file_clean:
        # if i.split('_')[0]+'_'+i.split('_')[1] != 'ecc09_2': # Comment here to run through all dataset. Right now it's set
        #     continue  #to ecc09 set for debugging.
        if (temp != (i.split('_')[0] + '_' + i.split('_')[1])) and (i[-5:] != 'clean'):
            temp = i.split('_')[0] + '_' + i.split('_')[1]
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            y3x = tot_file_clean[temp + '_y3x']
            y3y = tot_file_clean[temp + '_y3y']
            yoyo3 = np.transpose(np.array([y3x, y3y]))
            ax.plot(y3x, y3y, '+')
            ax.set_title(temp)
            plt.show()
            n_clusters = input("How many clusters are inside the plot? Please insert: ")
            plt.close(fig)
            km = KMeans(n_clusters=int(n_clusters), n_init=n_init, max_iter=max_iter, tol=tol, precompute_distances='auto',
                        verbose=0, random_state=None, algorithm='full')
            km.fit(yoyo3)
            kmeans[temp] = km
    #### shift the data to zero ####
    temp = ''
    tot_file_shift = {}
    for i in tot_file_clean:
        # if i.split('_')[0] + '_' + i.split('_')[1] != 'ecc09_2': # Comment here to run through all dataset. Right now it's set
        #     continue                   # to ecc09 set for debugging.
        if (temp != (i.split('_')[0] + '_' + i.split('_')[1])) and (i[-5:] != 'clean'):
            temp = i.split('_')[0] + '_' + i.split('_')[1]
            y1x_centered = np.zeros(len(handle.tot_file_clean[temp + '_y1x']))# centered data container.
            y1y_centered = np.zeros(len(handle.tot_file_clean[temp + '_y1y']))
            y3x_centered = np.zeros(len(handle.tot_file_clean[temp + '_y3x']))
            y3y_centered = np.zeros(len(handle.tot_file_clean[temp + '_y3y']))
            for j in range(kmeans[temp].n_clusters):
                center = kmeans[temp].cluster_centers_[j]
                mask = (kmeans[temp].labels_- j) == 0
                y1x = handle.tot_file_clean[temp + '_y1x'] #raw data
                y1y = handle.tot_file_clean[temp + '_y1y']
                y3x = handle.tot_file_clean[temp + '_y3x']
                y3y = handle.tot_file_clean[temp + '_y3y']
                y1x_ma = y1x[mask] # Masked data.Belonging to different group
                y1y_ma = y1y[mask]
                y3x_ma = y3x[mask]
                y3y_ma = y3y[mask]
                y1x_centered[mask] = y1x_ma - center[0] # Centered data. All shifted to zero
                y1y_centered[mask] = y1y_ma - center[1]
                y3x_centered[mask] = y3x_ma - center[0]
                y3y_centered[mask] = y3y_ma - center[1]

            tot_file_shift[temp + '_y1x'] = list(y1x_centered)
            tot_file_shift[temp + '_y1y'] = list(y1y_centered)
            tot_file_shift[temp + '_y3x'] = list(y3x_centered)
            tot_file_shift[temp + '_y3y'] = list(y3y_centered)
    os.chdir(handle.main_path+'/data')
    import json
    json = json.dumps(tot_file_shift)
    f = open('tot_file_clean.json', 'w')
    f.write(json)
    f.close()
    for filename in tot_file_shift:
        tot_file_shift[filename] = np.array(tot_file_shift[filename])
    handle.tot_file_shift = tot_file_shift
    return handle, tot_file_shift
def bashroi(handle):
    ## Manual clean all the data outside the ROI. This will pop up a window and let usr select the ROI.
    ## Data outside ROI will be deleted.
    ## Require bashvector.
    ## TODO:Implement here
    main_path = handle.main_path
    class ClickCap:
        def __init__(self, fig):
            self.xs = []
            self.ys = []
            self.times = 0
            self.cid = fig.canvas.mpl_connect('button_press_event', self)
        def __call__(self, event):
            if self.times != 1:
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
                print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                      ('double' if event.dblclick else 'single', event.button,
                       event.x, event.y, event.xdata, event.ydata))
                self.times = self.times + 1
            else:
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
                print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                      ('double' if event.dblclick else 'single', event.button,
                       event.x, event.y, event.xdata, event.ydata))
                self.times = 0
                plt.close('all')

    tot_vector = handle.tot_vector
    temp = ''
    roi = {}
    for i in tot_vector:
        if temp != i.split('_')[0] + '_' + i.split('_')[1]:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            cp = ClickCap(fig)
            temp = i.split('_')[0] + '_' + i.split('_')[1]
            ax.plot(tot_vector[temp + '_delx'], tot_vector[temp + '_dely'], '+')
            ax.set_xlim([-15, 15])
            ax.set_ylim([-12, 12])
            ax.set_title(temp+'_ROI selection')
            ax.grid(b=True, which='both', axis='both')
            plt.show()
            roi[temp + '_x'] = cp.xs
            roi[temp + '_y'] = cp.ys
            # print(crop)
    import json
    os.chdir(main_path+'/data')
    json = json.dumps(roi)
    f = open('roi.json', 'w')
    f.write(json)
    f.close()
    handle.roi = roi
    print(roi)
    return handle, roi
def bashclean(handle):
    tot_vector = handle.tot_vector
    maskfile = {}
    try:
        ## load roi file saved in ./roi/roi.json
        main_path = handle.main_path
        os.chdir(main_path + '/data')
        file = os.listdir()
        tot_file_clean ={}
        for i in file:
            if i != 'roi.json':
                continue
            else:
                roi = json.load(open(i))
                # print(roi)
                temp = ''
                for j in roi:
                    if temp != j.split('_')[0] + '_' + j.split('_')[1]:
                        temp = j.split('_')[0] + '_' + j.split('_')[1]
                        xmin = min(roi[temp + '_x'])
                        xmax = max(roi[temp + '_x'])
                        ymin = min(roi[temp + '_y'])
                        ymax = max(roi[temp + '_y'])
                        delx = tot_vector[temp + '_delx']
                        dely = tot_vector[temp + '_dely']
                        maskx = (delx > xmin) * (delx < xmax)
                        masky = (dely > ymin) * (dely < ymax)
                        mask = maskx * masky
                        handle.tot_vector[temp + '_delx_clean'] = tot_vector[temp + '_delx'][mask]
                        handle.tot_vector[temp + '_dely_clean'] = tot_vector[temp + '_dely'][mask]
                        tot_file_clean[temp + '_y1x'] = handle.tot_file[temp + '_y1x'][mask]
                        tot_file_clean[temp + '_y1y'] = handle.tot_file[temp + '_y1y'][mask]
                        tot_file_clean[temp + '_y3x'] = handle.tot_file[temp + '_y3x'][mask]
                        tot_file_clean[temp + '_y3y'] = handle.tot_file[temp + '_y3y'][mask]
                        maskfile[temp + '_mask'] = mask #False is the index of the outside points.
                handle.tot_file_clean = tot_file_clean
                handle.tot_roimask = maskfile
                print('Cleaning finished!')
        return handle, tot_file_clean
    except:
        print('Please run bashroi first.')
        return
def densitycal(handle, dataset = 'position', bins = 10, area = 3.14, x=[], y=[],ecc=0, debug='False'):
    # Calculate the density distribution in function of both radius and angle. Data clean and shift is required.
    # dataset = 'vec': vector representation, 'pos': position representation. bins: bin number.
    # debug: Testing mode. When turn to 'True', the function will accept x(np array) and y(np array) as the calculation data.
    # area: ellipse area(um^2). It shows the generic size of ellipse
    # No ellipse fitting in the end. Hard to justify if there is any mismatch.

    x0 = [15, 15]
    if debug == 'False':
        if dataset == 'position':
            data = handle.tot_pos_overlay_shift
        elif dataset =='vector':
            data = handle.tot_vec_overlay_clean

        fils = 6.25**2 # Filter intensity
        area = area*fils # convert unit to pixel^2. Roughly. The size of the filter ellipse is also changed here.

        temp = ''
        if 'del' in list(data.keys())[0]:
            density_hist = {}
            for filename in data:
                if temp != filename.split('_')[0]:
                    temp = filename.split('_')[0]
                    ecc = temp.replace('ecc', '')
                    if ecc == '0':
                        ecc = 0
                    else:
                        ecc = float('0.'+ecc.replace('0', ''))
                    delx = data[temp+'_delx']
                    dely = data[temp + '_dely']
                    x = np.power(delx, 2)
                    y = np.power(dely, 2)
                    a = (area**2/(4*(1-ecc**2)))**(0.25) # rough half-long axis
                    b = area/2/a # rough half-short axis
                    r = x/(a**2)+y/(b**2)
                    mask = r < 1
                    # TODO: Vector cut. Strength of the filter is set by the variable "fils" above. Theoretically it should be 6.25**2.
                    # Microscope pixel size calibration is needed.
                    r_edge = np.linspace(0, 1, bins)
                    d_edge = np.linspace(-np.pi, np.pi, bins)
                    x = delx[mask] #Actual data
                    y = dely[mask]
                    deg = np.arctan2(y, x)
                    r = x**2/(a**2)+y**2/(b**2)
                    area_e = np.zeros(len(r_edge)) #elliptical ring area
                    area_r = np.zeros(len(d_edge))#elliptical pizza area
                    for i in range(len(r_edge)):
                        if i == 0:
                            area_e[i] = 2*a*b*r_edge[0]
                        else:
                            area_e[i] = 2 * a * b * (r_edge[i]-r_edge[i-1])
                    density = np.zeros(len(area_e))
                    for i in range(len(r_edge)):
                        if i == 0:
                            r0 = 0
                        else:
                            r0 = r_edge[i-1]
                        r1 = r_edge[i]
                        density[i] = (np.sum((r<r1)*(r>r0)))/area_e[i]
                    rfunc = lambda theta: np.sqrt(1/(np.cos(theta)**2/a**2+np.sin(theta)**2/b**2))
                    deg_density = np.zeros(len(d_edge))
                    for j in range(len(d_edge)):
                        if j == 0:
                            continue
                        else:
                            d0 = d_edge[j-1]
                        d1 = d_edge[j]
                        area_r[j] = quad(rfunc, a=d0, b=d1)[0]
                        deg_density[j] = np.sum((deg>d0)*(deg<d1))/area_r[j]
                    density_hist[temp+'_density'] = density
                    density_hist[temp+'_edge'] = r_edge
                    density_hist[temp + '_degedge'] = d_edge
                    density_hist[temp + '_degdensity'] = deg_density
            handle.tot_density_hist = density_hist
        else:
            density_hist = {}
            for filename in data:
                if temp != filename.split('_')[0]:
                    temp = filename.split('_')[0]
                    y1x = data[temp+'_y1x']
                    y1y = data[temp + '_y1y']
                    y3x = data[temp + '_y3x']
                    y3y = data[temp + '_y3y']
                    ecc = temp.replace('ecc', '')
                    if ecc == '0':
                        ecc = 0
                    else:
                        ecc = float('0.'+ecc.replace('0', ''))

                    a = (area**2/(4*(1-ecc**2)))**(0.25) # rough half-long axis
                    b = area/2/a # rough half-short axis

                    r1 = y1x**2/(a**2)+y1y**2/(b**2)
                    r3 = y3x ** 2 / (a ** 2) + y3y ** 2 / (b ** 2)
                    mask1 = r1 < 1
                    mask3 = r3 < 1
                    # TODO: Vector cut. Strength of the filter is set by the variable "fils" above. Theoretically it should be 6.25**2.
                    # Microscope pixel size calibration is needed.
                    r_edge = np.linspace(0, 1, bins)
                    d_edge = np.linspace(-np.pi, np.pi, bins)

                    x1 = y1x[mask1]
                    y1 = y1y[mask1]
                    x3 = y3x[mask3]
                    y3 = y3y[mask3]

                    deg1 = np.arctan2(y1, x1)
                    deg3 = np.arctan2(y3, x3)
                    r1 = x1**2/(a**2)+y1**2/(b**2)
                    r3 = x3 ** 2 / (a ** 2) + y3 ** 2 / (b ** 2)
                    area_e = np.zeros(len(r_edge))
                    area_r = np.zeros(len(d_edge))
                    for i in range(len(r_edge)):
                        if i == 0:
                            area_e[i] = 2*a*b*r_edge[0]
                        else:
                            area_e[i] = 2 * a * b * (r_edge[i]-r_edge[i-1])

                    rfunc = lambda theta: np.sqrt(1 / (np.cos(theta) ** 2 / a ** 2 + np.sin(theta) ** 2 / b ** 2))
                    deg_density1 = np.zeros(len(d_edge))
                    deg_density3 = np.zeros(len(d_edge))
                    for j in range(len(d_edge)):
                        if j == 0:
                            continue
                        else:
                            d0 = d_edge[j - 1]
                        d1 = d_edge[j]
                        area_r[j] = quad(rfunc, a=d0, b=d1)[0]
                        deg_density1[j] = np.sum((deg1 > d0) * (deg1 < d1)) / area_r[j]
                        deg_density3[j] = np.sum((deg3 > d0) * (deg3 < d1)) / area_r[j]

                    density1 = np.zeros(len(area_e))
                    density3 = np.zeros(len(area_e))
                    for i in range(len(r_edge)):
                        if i == 0:
                            rin = 0
                        else:
                            rin = r_edge[i-1]
                        rout = r_edge[i]
                        density1[i] = (np.sum((r1 < rout)*(r1 > rin))) / area_e[i]
                        density3[i] = (np.sum((r3 < rout) * (r3 > rin))) / area_e[i]
                    density_hist[temp+'_density_y1'] = density1
                    density_hist[temp + '_density_y3'] = density3
                    density_hist[temp+'_edge'] = r_edge
                    density_hist[temp + '_degdensity_y1'] = deg_density1
                    density_hist[temp + '_degdensity_y3'] = deg_density3
                    density_hist[temp + '_degedge'] = d_edge
            handle.tot_density_hist = density_hist
    elif debug=='True':
        print('Debug mode on. The section here is purely for debugging purpose(backdoor).')
        fils = 6.25**2
        area = area*fils
        density_hist = {}
        delx = x
        dely = y
        x = np.power(delx, 2)
        y = np.power(dely, 2)
        a = (area ** 2 / (4 * (1 - ecc ** 2))) ** (0.25)  # rough half-long axis
        b = area / 2 / a  # rough half-short axis
        r = x / (a ** 2) + y / (b ** 2)
        mask = r < 1
        # TODO: Vector cut. Strength of the filter is set by the variable "fils" above. Theoretically it should be 6.25**2.
        # Microscope pixel size calibration is needed.
        r_edge = np.linspace(0, 1, bins)
        d_edge = np.linspace(-np.pi, np.pi, bins)
        x = delx[mask]  # Actual data
        y = dely[mask]
        deg = np.arctan2(y, x)
        r = x ** 2 / (a ** 2) + y ** 2 / (b ** 2)
        area_e = np.zeros(len(r_edge))  # elliptical ring area
        area_r = np.zeros(len(d_edge))  # elliptical pizza area
        for i in range(len(r_edge)):
            if i == 0:
                area_e[i] = 2 * a * b * r_edge[0]
            else:
                area_e[i] = 2 * a * b * (r_edge[i] - r_edge[i - 1])
        density = np.zeros(len(area_e))
        for i in range(len(r_edge)):
            if i == 0:
                r0 = 0
            else:
                r0 = r_edge[i - 1]
            r1 = r_edge[i]
            density[i] = (np.sum((r < r1) * (r > r0))) / area_e[i]
        rfunc = lambda theta: np.sqrt(1 / (np.cos(theta) ** 2 / a ** 2 + np.sin(theta) ** 2 / b ** 2))
        deg_density = np.zeros(len(d_edge))
        for j in range(len(d_edge)):
            if j == 0:
                continue
            else:
                d0 = d_edge[j - 1]
            d1 = d_edge[j]
            area_r[j] = quad(rfunc, a=d0, b=d1)[0]
            deg_density[j] = np.sum((deg > d0) * (deg < d1)) / area_r[j]
        density_hist['test_density'] = density
        density_hist['test_edge'] = r_edge
        density_hist['test_degedge'] = d_edge
        density_hist['test_degdensity'] = deg_density
    handle.tot_density_hist = density_hist
    return handle

####################### Simple theory module ###################################
# The functions listed here are to test different theoretical models and will be modified later.
def dhl(concentration = 1, ph = 7.5):
    # Estimate debye length of tris buffer. Debye-Huckel equation is applied.
    avo = Avogadro # Avogadro constant
    e = elementary_charge # Electron charge
    tris_con = 10*10**(-3)*concentration # tris concentration mol/L
    h_con = 10**(-ph) # H+ concentration mol/l
    oh_con = 10**(-14)/h_con # OH- concentration mol/L
    pka = 8.07 # from https://en.wikipedia.org/wiki/Tris


    tris_conjugate = tris_con/((10**(-pka)/(h_con))+1)
    tris_ini = tris_con - tris_conjugate
    cl_con = tris_conjugate+h_con-oh_con
    epsilon = 78 # Permittivity of 1X solution. Permittivity decreases linearly when the salt concentration is less than 1M.
    t = 298 # 25celcius
    def mol2n(x):
        return avo*x*1000*(e**2)
    return np.sqrt((epsilon_0*epsilon*Boltzmann*t)
                   / (mol2n(tris_conjugate)
                    + mol2n(h_con)
                      +mol2n(oh_con)
                      + mol2n(cl_con)))
def bulk_fe(alpha=1, beta = 1, n=10):
    # calculate the Flory eneryg:  van der Waals model + entropical energy.
    # According to equ. (21)(22) in Walter's review paper.
    # Input: alpha, prefactor describing the internal property of the chain. To be decided with experiment.
    #       beta: prefactor describing the internal propertyy of the chain.
    #       Comment: internal property includes: width, persist-length, temperature(explicitly) and so on(implicitly)
    #       n: number of segments for one chain.
    #       r: size of the chain
    return alpha*n**(-1/5)+beta*n**(1/5)
def degenn_fe(alpha=1, l=1, D=1):
    # Calculate degenn free energy in nanochannel.
    # Reference: Walter's review paper
    # Input: l, size of the chain. Rg or end-to-end length.
    #       D, effective diameter of the nanochannel.
    return alpha*l/D**5/3
def wall_depletion(r=0.1):
    # Input:    r, depletion ratio
    #       alpha, intensity of van der waals extraction
    # Output:   concentration ratio
    if r >= 1:
        a = 1
    else:
        a = r**(5/3)
    return a
def blobsize():
    # Calculate the blob-size of polymer in cavity
    # Walter suggested to treat polymer as solution.

    return
def ellipse_para(ecc=0):
    # output: width in um
    area = 3.14
    a = np.sqrt(area**2/(np.pi**2*(1-ecc**2)))
    b = area/np.pi/a
    return [a, b]
def ellipse_width(ecc=0):
    # output: x, x-position in um
    #       y, y-position in um
    [a, b] = ellipse_para(ecc)
    x = np.linspace(-a, a, 200)
    y = np.sqrt((1 - x**2/a**2) * b**2)
    return x, y

# TODO: Implement van der waals potential to wall depletion.
###############################################
if __name__=="__main__":
    # for ecc=0.8
    ecc = 0.8
    depth = 0.2 # depth in um
    x, y = ellipse_width(ecc)
    d_eff = np.sqrt(2*y*depth) # effective width

    plt.plot(x, d_eff)
    plt.show()

