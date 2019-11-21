import numpy as np
import matplotlib.pyplot as plt
import module
import json
import os
import scipy.stats as stats
from matplotlib.lines import Line2D
main_path = "/media/zezhou/Seagate Expansion Drive/McGillResearch/2019Manuscript_Analysis/Analysis/datafterlinearshift/tplasmid"
cleanmode = 1 # 1-cleaned data. The roi.json and tot_file_clean.json files have been saved in ./data folder.
              # 0-raw data. Experimental data before shift to zero.
# Read files
handle1, tot_file = module.bashload(main_path)
handle1, tot_vector = module.bashvector(handle1)
handle1, tot_vec_overlay = module.bashoverlay(handle1)

# Data clean
if cleanmode == 0:
    handle1, roi = module.bashroi(handle1) # ROI selection
    handle1, tot_file_clean = module.bashclean(handle1) # Delete points ouside ROI and attach mask to handle1.
    handle1, tot_file_shift = module.bashshift(handle1) # Shift data to zero according to YOYO-3 channel
elif cleanmode == 1:
    os.chdir(main_path+'/data')
    tot_file_clean = json.load(open('tot_file_clean.json')) # data is saved in list format
    for filename in tot_file_clean:
        tot_file_clean[filename] = np.array(tot_file_clean[filename]) # Changing format to array
    handle1.tot_file_shift = tot_file_clean

# Cleaned data re-calculate
handle1, tot_vector_clean = module.bashvector(handle1, mode='clean')
handle1, tot_vec_overlay_clean = module.bashoverlay(handle1, mode='clean')
handle1, tot_pos_overlay_shift = module.bashoverlay(handle1, mode='clean', set='position')

# Calculate correlation

# Initialize setting/plot parameters
t = 35 # entire correlation frames
fps = 17.5
titlefontsize = 20
labelsize = 15
cb_fontsize = 15
ticksize = 15
figuresize = 20
avg_frames = 500
savepath = '/media/zezhou/Seagate Expansion Drive/McGillResearch/2019Manuscript_Analysis/Analysis/Plots/tplasmid/autocorr/'
# ecc0
def corr_fix(corr):
    m = len(corr)
    u = np.arange(1, (m + 1) / 2 + 1)
    n = np.concatenate((u, np.arange((m + 1) / 2 - 1, 0, -1)))
    return corr/n
def ecc0(frames):
    frames = avg_frames
    y1x = []
    y1y = []
    y3x = []
    y3y = []
    tot = np.floor_divide(len(handle1.tot_file_shift['ecc0_1_y1x']), frames)
    for i in range(tot):
        y1x.append(handle1.tot_file_shift['ecc0_1_y1x'][i*frames:(i+1)*frames])
        y1y.append(handle1.tot_file_shift['ecc0_1_y1y'][i*frames:(i+1)*frames])
        y3x.append(handle1.tot_file_shift['ecc0_1_y3x'][i*frames:(i+1)*frames])
        y3y.append(handle1.tot_file_shift['ecc0_1_y3y'][i*frames:(i+1)*frames])
    return y1x, y1y, y3x, y3y
def ecc0auto():
    y1x, y1y, y3x, y3y = ecc0(avg_frames)
    mag = 1
    a_y1x = []
    a_y1y = []
    a_y3x = []
    a_y3y = []
    frames = len(y1x[0])
    for i in range(len(y1x)):
        if i == 2:
            continue
        a = corr_fix(np.correlate(y1x[i], y1x[i], mode='full'))
        b = corr_fix(np.correlate(y1y[i], y1y[i], mode='full'))
        c = corr_fix(np.correlate(y3x[i], y3x[i], mode='full'))
        d = corr_fix(np.correlate(y3y[i], y3y[i], mode='full'))
        a_y1x.append(a/a[frames]*mag)
        a_y1y.append(b/b[frames]*mag)
        a_y3x.append(c/c[frames]*mag)
        a_y3y.append(d/d[frames]*mag)
    a_y1x = np.array(a_y1x)
    a_y1y = np.array(a_y1y)
    a_y3x = np.array(a_y3x)
    a_y3y = np.array(a_y3y)
    return a_y1x, a_y1y, a_y3x, a_y3y

def ecc06(frames):
    frames = avg_frames
    y1x = []
    y1y = []
    y3x = []
    y3y = []
    iter = 0
    for item in handle1.tot_file_shift:
        if 'ecc06' in item:
            iter +=1
    iter = int(iter/4)
    for k in range(iter):
        y1x_tmp = 'ecc06_' + str(k + 1) + '_y1x'
        y1y_tmp = 'ecc06_' + str(k + 1) + '_y1y'
        y3x_tmp = 'ecc06_' + str(k + 1) + '_y3x'
        y3y_tmp = 'ecc06_' + str(k + 1) + '_y3y'
        tot = np.floor_divide(len(handle1.tot_file_shift[y1x_tmp]), frames)
        for i in range(tot):
            y1x.append(handle1.tot_file_shift[y1x_tmp][i*frames:(i+1)*frames])
            y1y.append(handle1.tot_file_shift[y1y_tmp][i*frames:(i+1)*frames])
            y3x.append(handle1.tot_file_shift[y3x_tmp][i*frames:(i+1)*frames])
            y3y.append(handle1.tot_file_shift[y3y_tmp][i*frames:(i+1)*frames])
    return y1x, y1y, y3x, y3y
def ecc06auto():
    y1x, y1y, y3x, y3y = ecc06(avg_frames)
    mag = 1
    a_y1x = []
    a_y1y = []
    a_y3x = []
    a_y3y = []
    frames = len(y1x[0])
    for i in range(len(y1x)):
        a = corr_fix(np.correlate(y1x[i], y1x[i], mode='full'))
        b = corr_fix(np.correlate(y1y[i], y1y[i], mode='full'))
        c = corr_fix(np.correlate(y3x[i], y3x[i], mode='full'))
        d = corr_fix(np.correlate(y3y[i], y3y[i], mode='full'))
        a_y1x.append(a / a[frames] * mag)
        a_y1y.append(b / b[frames] * mag)
        a_y3x.append(c / c[frames] * mag)
        a_y3y.append(d / d[frames] * mag)
    a_y1x = np.array(a_y1x)
    a_y1y = np.array(a_y1y)
    a_y3x = np.array(a_y3x)
    a_y3y = np.array(a_y3y)
    return a_y1x, a_y1y, a_y3x, a_y3y

def ecc08(frames):
    frames = avg_frames
    y1x = []
    y1y = []
    y3x = []
    y3y = []
    iter = 0
    for item in handle1.tot_file_shift:
        if 'ecc08' in item:
            iter += 1
    iter = int(iter / 4)
    for k in range(iter):
        y1x_tmp = 'ecc08_' + str(k + 1) + '_y1x'
        y1y_tmp = 'ecc08_' + str(k + 1) + '_y1y'
        y3x_tmp = 'ecc08_' + str(k + 1) + '_y3x'
        y3y_tmp = 'ecc08_' + str(k + 1) + '_y3y'
        tot = np.floor_divide(len(handle1.tot_file_shift[y1x_tmp]), frames)
        for i in range(tot):
            y1x.append(handle1.tot_file_shift[y1x_tmp][i * frames:(i + 1) * frames])
            y1y.append(handle1.tot_file_shift[y1y_tmp][i * frames:(i + 1) * frames])
            y3x.append(handle1.tot_file_shift[y3x_tmp][i * frames:(i + 1) * frames])
            y3y.append(handle1.tot_file_shift[y3y_tmp][i * frames:(i + 1) * frames])
    return y1x, y1y, y3x, y3y
def ecc08auto():
    y1x, y1y, y3x, y3y = ecc08(avg_frames)
    mag = 1
    a_y1x = []
    a_y1y = []
    a_y3x = []
    a_y3y = []
    frames = len(y1x[0])
    for i in range(len(y1x)):
        a = corr_fix(np.correlate(y1x[i], y1x[i], mode='full'))
        b = corr_fix(np.correlate(y1y[i], y1y[i], mode='full'))
        c = corr_fix(np.correlate(y3x[i], y3x[i], mode='full'))
        d = corr_fix(np.correlate(y3y[i], y3y[i], mode='full'))
        a_y1x.append(a / a[frames] * mag)
        a_y1y.append(b / b[frames] * mag)
        a_y3x.append(c / c[frames] * mag)
        a_y3y.append(d / d[frames] * mag)
    a_y1x = np.array(a_y1x)
    a_y1y = np.array(a_y1y)
    a_y3x = np.array(a_y3x)
    a_y3y = np.array(a_y3y)
    return a_y1x, a_y1y, a_y3x, a_y3y

def ecc09(frames):
    frames = avg_frames
    y1x = []
    y1y = []
    y3x = []
    y3y = []
    iter = 0
    for item in handle1.tot_file_shift:
        if 'ecc09_' in item:
            iter += 1
    iter = int(iter / 4)
    for k in range(iter):
        y1x_tmp = 'ecc09_' + str(k + 1) + '_y1x'
        y1y_tmp = 'ecc09_' + str(k + 1) + '_y1y'
        y3x_tmp = 'ecc09_' + str(k + 1) + '_y3x'
        y3y_tmp = 'ecc09_' + str(k + 1) + '_y3y'
        tot = np.floor_divide(len(handle1.tot_file_shift[y1x_tmp]), frames)
        for i in range(tot):
            y1x.append(handle1.tot_file_shift[y1x_tmp][i * frames:(i + 1) * frames])
            y1y.append(handle1.tot_file_shift[y1y_tmp][i * frames:(i + 1) * frames])
            y3x.append(handle1.tot_file_shift[y3x_tmp][i * frames:(i + 1) * frames])
            y3y.append(handle1.tot_file_shift[y3y_tmp][i * frames:(i + 1) * frames])
    return y1x, y1y, y3x, y3y
def ecc09auto():
    y1x, y1y, y3x, y3y = ecc09(avg_frames)
    mag = 1
    a_y1x = []
    a_y1y = []
    a_y3x = []
    a_y3y = []
    frames = len(y1x[0])
    for i in range(len(y1x)):
        a = corr_fix(np.correlate(y1x[i], y1x[i], mode='full'))
        b = corr_fix(np.correlate(y1y[i], y1y[i], mode='full'))
        c = corr_fix(np.correlate(y3x[i], y3x[i], mode='full'))
        d = corr_fix(np.correlate(y3y[i], y3y[i], mode='full'))
        # print(i)
        a_y1x.append(a / a[frames] * mag)
        a_y1y.append(b / b[frames] * mag)
        a_y3x.append(c / c[frames] * mag)
        a_y3y.append(d / d[frames] * mag)
    a_y1x = np.array(a_y1x)
    a_y1y = np.array(a_y1y)
    a_y3x = np.array(a_y3x)
    a_y3y = np.array(a_y3y)
    return a_y1x, a_y1y, a_y3x, a_y3y

def ecc095(frames):
    frames = avg_frames
    y1x = []
    y1y = []
    y3x = []
    y3y = []
    iter = 0
    for item in handle1.tot_file_shift:
        if 'ecc095_' in item:
            iter += 1
    iter = int(iter / 4)
    for k in range(iter):
        y1x_tmp = 'ecc095_' + str(k + 1) + '_y1x'
        y1y_tmp = 'ecc095_' + str(k + 1) + '_y1y'
        y3x_tmp = 'ecc095_' + str(k + 1) + '_y3x'
        y3y_tmp = 'ecc095_' + str(k + 1) + '_y3y'
        tot = np.floor_divide(len(handle1.tot_file_shift[y1x_tmp]), frames)
        for i in range(tot):
            y1x.append(handle1.tot_file_shift[y1x_tmp][i * frames:(i + 1) * frames])
            y1y.append(handle1.tot_file_shift[y1y_tmp][i * frames:(i + 1) * frames])
            y3x.append(handle1.tot_file_shift[y3x_tmp][i * frames:(i + 1) * frames])
            y3y.append(handle1.tot_file_shift[y3y_tmp][i * frames:(i + 1) * frames])
    return y1x, y1y, y3x, y3y
def ecc095auto():
    y1x, y1y, y3x, y3y = ecc095(avg_frames)
    mag = 1
    a_y1x = []
    a_y1y = []
    a_y3x = []
    a_y3y = []
    frames = len(y1x[0])
    for i in range(len(y1x)):
        a = corr_fix(np.correlate(y1x[i], y1x[i], mode='full'))
        b = corr_fix(np.correlate(y1y[i], y1y[i], mode='full'))
        c = corr_fix(np.correlate(y3x[i], y3x[i], mode='full'))
        d = corr_fix(np.correlate(y3y[i], y3y[i], mode='full'))
        # print(i)
        a_y1x.append(a / a[frames] * mag)
        a_y1y.append(b / b[frames] * mag)
        a_y3x.append(c / c[frames] * mag)
        a_y3y.append(d / d[frames] * mag)
    a_y1x = np.array(a_y1x)
    a_y1y = np.array(a_y1y)
    a_y3x = np.array(a_y3x)
    a_y3y = np.array(a_y3y)
    return a_y1x, a_y1y, a_y3x, a_y3y

def ecc098(frames):
    frames = avg_frames
    y1x = []
    y1y = []
    y3x = []
    y3y = []
    iter = 0
    for item in handle1.tot_file_shift:
        if 'ecc098_' in item:
            iter += 1
    iter = int(iter / 4)+1
    for k in range(iter):
        if k == 6:
            continue
        y1x_tmp = 'ecc098_' + str(k + 1) + '_y1x'
        y1y_tmp = 'ecc098_' + str(k + 1) + '_y1y'
        y3x_tmp = 'ecc098_' + str(k + 1) + '_y3x'
        y3y_tmp = 'ecc098_' + str(k + 1) + '_y3y'
        tot = np.floor_divide(len(handle1.tot_file_shift[y1x_tmp]), frames)
        for i in range(tot):
            y1x.append(handle1.tot_file_shift[y1x_tmp][i * frames:(i + 1) * frames])
            y1y.append(handle1.tot_file_shift[y1y_tmp][i * frames:(i + 1) * frames])
            y3x.append(handle1.tot_file_shift[y3x_tmp][i * frames:(i + 1) * frames])
            y3y.append(handle1.tot_file_shift[y3y_tmp][i * frames:(i + 1) * frames])
    return y1x, y1y, y3x, y3y
def ecc098auto():
    y1x, y1y, y3x, y3y = ecc098(avg_frames)
    mag = 1
    a_y1x = []
    a_y1y = []
    a_y3x = []
    a_y3y = []
    frames = len(y1x[0])
    for i in range(len(y1x)):
        a = corr_fix(np.correlate(y1x[i], y1x[i], mode='full'))
        b = corr_fix(np.correlate(y1y[i], y1y[i], mode='full'))
        c = corr_fix(np.correlate(y3x[i], y3x[i], mode='full'))
        d = corr_fix(np.correlate(y3y[i], y3y[i], mode='full'))
        a_y1x.append(a / a[frames] * mag)
        a_y1y.append(b / b[frames] * mag)
        a_y3x.append(c / c[frames] * mag)
        a_y3y.append(d / d[frames] * mag)
    a_y1x = np.array(a_y1x)
    a_y1y = np.array(a_y1y)
    a_y3x = np.array(a_y3x)
    a_y3y = np.array(a_y3y)
    return a_y1x, a_y1y, a_y3x, a_y3y

def ecc0995(frames):
    frames = avg_frames
    y1x = []
    y1y = []
    y3x = []
    y3y = []
    iter = 0
    for item in handle1.tot_file_shift:
        if 'ecc0995_' in item:
            iter += 1
    iter = int(iter / 4)
    for k in range(iter):
        y1x_tmp = 'ecc0995_' + str(k + 1) + '_y1x'
        y1y_tmp = 'ecc0995_' + str(k + 1) + '_y1y'
        y3x_tmp = 'ecc0995_' + str(k + 1) + '_y3x'
        y3y_tmp = 'ecc0995_' + str(k + 1) + '_y3y'
        tot = np.floor_divide(len(handle1.tot_file_shift[y1x_tmp]), frames)
        for i in range(tot):
            y1x.append(handle1.tot_file_shift[y1x_tmp][i * frames:(i + 1) * frames])
            y1y.append(handle1.tot_file_shift[y1y_tmp][i * frames:(i + 1) * frames])
            y3x.append(handle1.tot_file_shift[y3x_tmp][i * frames:(i + 1) * frames])
            y3y.append(handle1.tot_file_shift[y3y_tmp][i * frames:(i + 1) * frames])
    return y1x, y1y, y3x, y3y
def ecc0995auto():
    y1x, y1y, y3x, y3y = ecc0995(avg_frames)
    mag = 1
    a_y1x = []
    a_y1y = []
    a_y3x = []
    a_y3y = []
    frames = len(y1x[0])
    for i in range(len(y1x)):
        a = corr_fix(np.correlate(y1x[i], y1x[i], mode='full'))
        b = corr_fix(np.correlate(y1y[i], y1y[i], mode='full'))
        c = corr_fix(np.correlate(y3x[i], y3x[i], mode='full'))
        d = corr_fix(np.correlate(y3y[i], y3y[i], mode='full'))
        # print(i)
        a_y1x.append(a / a[frames] * mag)
        a_y1y.append(b / b[frames] * mag)
        a_y3x.append(c / c[frames] * mag)
        a_y3y.append(d / d[frames] * mag)
    a_y1x = np.array(a_y1x)
    a_y1y = np.array(a_y1y)
    a_y3x = np.array(a_y3x)
    a_y3y = np.array(a_y3y)
    return a_y1x, a_y1y, a_y3x, a_y3y
auto = [ecc0auto(), ecc06auto(), ecc08auto(), ecc09auto(),
        ecc095auto(), ecc098auto(),ecc0995auto()] # Entire data set we want to include
ndata = len(auto)
#### Parameters of plotting ####
# starting frame
p_fit_frames=[13,13,13,13,13,13,5] # exp fitting start frames:plasmid/y1
t_fit_frames=[3,3,3,8,8,9,12] # exp fitting start frames:t4/y3

sample_frames_offset_y1=[0,0,0,0,0,13,0] # Sample frames offset/p
sample_frames_offset_y3=[0,10,15,35,0,10,5]
marker_tot = ['Eccentricity=0', 'Eccentricity=0.6', 'Eccentricity=0.8',
              'Eccentricity=0.9', 'Eccentricity=0.95', 'Eccentricity=0.98', 'Eccentricity=0.995']
# a_y1x, a_y1y, a_y3x, a_y3y = ecc0995auto() # Dataset setting
fit_para_y1x =[]
fit_para_y3x =[]
#### Plot correlation ####
fig = plt.figure(figsize=[figuresize, figuresize/1.3333])
for i in range(ndata):
    ax2 = fig.add_subplot(3,3,i+1)
    a_y1x, a_y1y, a_y3x, a_y3y = auto[i]
    marker = marker_tot[i]
    # Data average and standard error of mean
    test = len(a_y1x[0, :])
    a_y1x = a_y1x[:, int((len(a_y1x[0, :])+1)/2):int((len(a_y1x[0, :])+1)/2+t)]
    m_y1x = np.mean(a_y1x, axis=0)
    e_y1x = stats.sem(a_y1x)

    a_y3x = a_y3x[:, int((len(a_y3x[0, :])+1)/2):int((len(a_y3x[0, :])+1)/2+t)]
    m_y3x = np.mean(a_y3x, axis=0)
    e_y3x = stats.sem(a_y3x)

    # generate x-axis(in second)
    x = np.arange(0, len(m_y1x))/17.5

    # generate line

    ax2.errorbar(x, m_y1x, yerr=e_y1x)
    ax2.errorbar(x, m_y3x, yerr=e_y3x)
    lines = ax2.get_lines()

    for j in range(5):
        line3 = ax2.plot(x, a_y1x[j+sample_frames_offset_y1[i],:], color = lines[0].get_color(), alpha=0.25, ls = '--')
    for j in range(5):
        line4 = ax2.plot(x, a_y3x[j+sample_frames_offset_y3[i],:], color = lines[1].get_color(), alpha=0.25, ls = '--')

    lines = ax2.get_lines()


    # exponential fitting
    fitx_y1x = x[p_fit_frames[i]:]
    fitx_y3x = x[t_fit_frames[i]:]
    fit_y1x = np.polyfit(fitx_y1x, np.log(m_y1x[p_fit_frames[i]:]), deg=int(1))
    fit_y3x = np.polyfit(fitx_y3x, np.log(m_y3x[t_fit_frames[i]:]), deg=int(1))
    fit_para_y1x.append(fit_y1x)
    fit_para_y3x.append(fit_y3x)
    # plot exponential fitting
    line_fit = ax2.plot(fitx_y1x, np.exp(fit_y1x[0]*fitx_y1x+fit_y1x[1]),'--', color = 'r')
    ax2.plot(fitx_y3x, np.exp(fit_y3x[0]*fitx_y3x+fit_y3x[1]),'--', color = 'r')


    #####################3
    ax2.set_yscale('symlog')
    xlim = t/fps
    xtick_temp = np.arange(0, xlim+0.5, step = 0.5)
    xticks = list(xtick_temp)
    xticklabel = []

    ytick_temp = np.logspace(-2, 0, 3)
    ax2.set_ylim(0.001, 1.1)
    yticks = list(ytick_temp)

    ax2.set_xticks(xticks)
    ax2.set_yticks(yticks)
    ax2.tick_params(axis='both', labelsize=ticksize)

    ax2.set_xlabel('Lag time(s)', fontsize=labelsize)
    ax2.set_ylabel('Position auto-correlation(a.u.)', fontsize=labelsize)
    # ax2.legend((lines[0], lines[1], line3[0], line4[0], line_fit[0]), ('Plasmid', 'T4-DNA',
    #                                                       'Sample of individual plasmids', 'Sample of individual T4-DNA',
    #                                                                 'Exponential fitting'),
    #            fontsize=labelsize)
    ax2.text(x=xlim * 0.6, y= 1, s=marker, fontsize=labelsize)
ax2.legend((lines[0], lines[1], line3[0], line4[0], line_fit[0]), ('Plasmid', 'T4-DNA',
                                                          'Sample of individual plasmids', 'Sample of individual T4-DNA',
                                                                    'Exponential fitting'),
               fontsize=labelsize, loc=3)
ax2 = fig.add_subplot(3,3,8)

## slope plot
tickslabel = ['0', '0', '0.6', '0.8', '0.9', '0.95', '0.98', '0.995']
slope_y3x = []
for item in fit_para_y3x:
    slope_y3x.append((-1/item[0]))
slope_y1x = []
for item in fit_para_y1x:
    slope_y1x.append((-1/item[0]))
ax2.plot(slope_y1x)
ax2.plot(slope_y3x)
ax2.set_xticklabels(tickslabel)
ax2.set_xlabel('Eccentricity', fontsize = labelsize)
ax2.set_ylabel('Relaxation time(s)', fontsize = labelsize)
ax2.legend(['Plasmid', 'T4-DNA'], fontsize=labelsize)
# os.chdir(savepath)
# plt.savefig('autocorr.png')
# plt.savefig('autocorr.eps')
# plt.savefig('autocorr.pdf')
plt.show()