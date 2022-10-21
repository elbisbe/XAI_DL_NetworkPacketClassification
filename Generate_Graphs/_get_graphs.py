import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import seaborn as sn
import matplotlib
matplotlib.use('agg')

from numpy import genfromtxt
import matplotlib.pyplot as plt

#### FIGURES
### 1 GRADCAM STANDARD VS COUNTERFACTUAL
### 2 INDEPENDENT FLOWS
### 3 DEPENDENT FLOWS
### 4 GRADCAM-2D
### 5 CONFUSION MATRIX
### 6 PIXEL PADDING DIFFERNECE
### 7 FTP UPLINK VS DOWNLINK
### 8 2D HISTOGRAM EMAIL
### 9 2D HISTOGRAM FTPS
### 10 REPRESENTATIVE VS ALL GRADCAM HISTOGRAMS
### 11 REPRESENTATIVE VS LENGTH HISTOGRAM


### 1 GRADCAM STANDARD VS COUNTERFACTUAL
print("### 1 GRADCAM STANDARD VS COUNTERFACTUAL")
standard_gradcam = genfromtxt("standard_gradcam.csv",delimiter=',')
counterfactual_gradcam = genfromtxt("counterfactual_gradcam.csv",delimiter=',')

plt.plot(standard_gradcam)
plt.plot(counterfactual_gradcam)

plt.xlabel("Length of packet (bytes)")
plt.ylabel("Relevance value of the byte in packet")
plt.title("Standard GradCAM 1D heatmap vs \ncounterfactual GradCAM 1D heatmap")
plt.legend(["Standard GradCAM","counterfactual GradCAM"], loc="upper right")
plt.savefig("GradCAM_standard_vs_counterfactual.pdf")

###########################################################################
### 2 INDEPENDENT FLOWS
plt.clf()
print("### 2 INDEPENDENT FLOWS")
training_independent = genfromtxt("training_independent.csv",delimiter=',')
validation_independent = genfromtxt("validation_independent.csv",delimiter=',')
x = np.linspace(0,50,50)
plt.plot(x,training_independent,"r*", markersize=5)
plt.plot(x,validation_independent,lw=3)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and validation \naccuracy on independent flows")
plt.legend(["Training accuracy","Validation accuracy"], loc="lower right")
plt.savefig("AccuracySubplot_dependent.pdf")

###########################################################################
### 3 DEPENDENT FLOWS
plt.clf()
print("### 3 DEPENDENT FLOWS")

training_dependent = genfromtxt("training_dependent.csv",delimiter=',')
validation_dependent = genfromtxt("validation_dependent.csv",delimiter=',')
x = np.linspace(0,50,51)
plt.plot(x,training_dependent,"r*", markersize=5)
plt.plot(x,validation_dependent,lw=3)

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and validation \naccuracy on dependent flows")
plt.legend(["Training accuracy","Validation accuracy"], loc="lower right")
plt.savefig("AccuracySubplot_independent.pdf")

###########################################################################
### 4 GRADCAM-2D
print("### 4 GRADCAM-2D")
print("example_2D heatmap.png")


###########################################################################
### 5 CONFUSION MATRIX
print("### 5 CONFUSION MATRIX")
plt.clf()

cm = genfromtxt("cm.csv",delimiter=',')
df_cm = pd.DataFrame(cm, index = ["Chat","Audio","Email","FTPS","Video"],
                  columns = ["Chat","Audio","Email","FTPS","Video"])
plt.figure(figsize = (10,7))
plt.tight_layout()
sn.heatmap(df_cm, annot=True,cbar_kws={'format': '%.0f%%'})
plt.savefig("Classification_Matrix.pdf",bbox_inches='tight')


###########################################################################
### 6 PIXEL PADDING DIFFERNECE
print("### 6 PIXEL PADDING DIFFERNECE")
plt.clf()

onepx_padding = genfromtxt("onepx_padding.csv",delimiter=',')
multpx_padding = genfromtxt("multpx_padding.csv",delimiter=',')

_ = plt.plot(onepx_padding, color="black")
_ = plt.plot(multpx_padding, "r-", markersize=2, lw=3)

plt.xlabel("Length of packet (bytes)")
plt.ylabel("Relevance value of the byte in packet")
plt.title("1px padding vs 2px+ padding in \nChat standard GradCAM heatmap")
plt.legend(["1px padding","2px+ padding"], loc="upper right")
plt.savefig("Padding.pdf")
plt.show()

###########################################################################
### 7 FTP UPLINK VS DOWNLINK
print("### 7 FTP UPLINK VS DOWNLINK")
plt.clf()

ftp_upload = genfromtxt("ftp_upload.csv",delimiter=',')
ftp_download = genfromtxt("ftp_download.csv",delimiter=',')

_ = plt.plot(ftp_upload)
_ = plt.plot(ftp_download)

plt.xlabel("Length of packet (bytes)")
plt.ylabel("Relevance value of the byte in packet")
plt.title("FTP uplink download traffic heatmap vs. \nFTP downlink download traffic heatmap")
plt.legend(["Uplink","Downlink"], loc="upper right")
plt.savefig('Client_Server.pdf') 

###########################################################################
### 8 2D HISTOGRAM EMAIL
print("### 8 2D HISTOGRAM EMAIL")
plt.clf()

email_2d_x = genfromtxt("email_2d_x.csv",delimiter=',')
email_2d_y = genfromtxt("email_2d_y.csv",delimiter=',')

fig = plt.figure(figsize = (6,6))
plt.hist2d(email_2d_x, email_2d_y, bins=32)
ax = fig.axes[0]

plt.xlabel("Relevance value of the byte in x axis")
plt.ylabel("Relevance value of the byte in y axis")
plt.title("2D Histogram of relevant bytes for all Email flows")
ax.invert_yaxis()
plt.savefig("Email 2D Email.pdf",bbox_inches='tight')
plt.show()

###########################################################################
### 9 2D HISTOGRAM FTPS
print("### 9 2D HISTOGRAM FTPS")
plt.clf()

ftps_2d_x = genfromtxt("ftps_2d_x.csv",delimiter=',')
ftps_2d_y = genfromtxt("ftps_2d_y.csv",delimiter=',')

fig = plt.figure(figsize = (6,6))
plt.hist2d(ftps_2d_x, ftps_2d_y, bins=32)
ax = fig.axes[0]

plt.xlabel("Relevance value of the byte in x axis")
plt.ylabel("Relevance value of the byte in y axis")
plt.title("2D Histogram of relevant bytes for all FTPS flows")
ax.invert_yaxis()
plt.savefig("Email 2D FTPS.pdf",bbox_inches='tight')
plt.show()


###########################################################################
### 10 REPRESENTATIVE VS ALL GRADCAM HISTOGRAMS
print("### 10 REPRESENTATIVE VS ALL GRADCAM HISTOGRAMS")
plt.clf()

fig, axs = plt.subplots(5)
fig.set_size_inches(6.5, 18)
#fig.suptitle('GradCAM + Histogramas')
plt.subplots_adjust(left=0.2,
                    bottom=0.1, 
                    right=1, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
#plt.tight_layout()

chat_len_hist = genfromtxt("chat_len_hist.csv",delimiter=',')
chat_func1 = genfromtxt("chat_func1.csv",delimiter=',')
chat_func2 = genfromtxt("chat_func2.csv",delimiter=',')

audio_len_hist = genfromtxt("audio_len_hist.csv",delimiter=',')
audio_func1 = genfromtxt("audio_func1.csv",delimiter=',')
audio_func2 = genfromtxt("audio_func2.csv",delimiter=',')

email_len_hist = genfromtxt("email_len_hist.csv",delimiter=',')
email_func1 = genfromtxt("email_func1.csv",delimiter=',')
email_func2 = genfromtxt("email_func2.csv",delimiter=',')

ftps_len_hist = genfromtxt("ftps_len_hist.csv",delimiter=',')
ftps_func1 = genfromtxt("ftps_func1.csv",delimiter=',')
ftps_func2 = genfromtxt("ftps_func2.csv",delimiter=',')

video_len_hist = genfromtxt("video_len_hist.csv",delimiter=',')
video_func1 = genfromtxt("video_func1.csv",delimiter=',')
video_func2 = genfromtxt("video_func2.csv",delimiter=',')

chat_gcam_hists_standar = genfromtxt("chat_gcam_hists_standar.csv",delimiter=',')
chat_gcam_hists_counter = genfromtxt("chat_gcam_hists_counter.csv",delimiter=',')

audio_gcam_hists_standar = genfromtxt("audio_gcam_hists_standar.csv",delimiter=',')
audio_gcam_hists_counter = genfromtxt("audio_gcam_hists_counter.csv",delimiter=',')

email_gcam_hists_standar = genfromtxt("email_gcam_hists_standar.csv",delimiter=',')
email_gcam_hists_counter = genfromtxt("email_gcam_hists_counter.csv",delimiter=',')

ftps_gcam_hists_standar = genfromtxt("ftps_gcam_hists_standar.csv",delimiter=',')
ftps_gcam_hists_counter = genfromtxt("ftps_gcam_hists_counter.csv",delimiter=',')

video_gcam_hists_standar = genfromtxt("video_gcam_hists_standar.csv",delimiter=',')
video_gcam_hists_counter = genfromtxt("video_gcam_hists_counter.csv",delimiter=',')

print("Processing Chat")

for i in range(len(chat_gcam_hists_standar)):
    axs[0].plot(chat_gcam_hists_counter[i,:],color="#ff9175", rasterized=True)
for i in range(len(chat_gcam_hists_counter)):
    axs[0].plot(chat_gcam_hists_standar[i,:],color="#4d9157", rasterized=True)
axs[0].plot(chat_func1, color="red", rasterized=True)
axs[0].plot(chat_func2, color="green", rasterized=True)
axs[0].set(xlabel="Length of packet (bytes)", ylabel="Relevance value of the byte in \nflow and mean of clusters (lines)", title="Chat packets representative functions vs all Chat histograms")

print("Processing Audio")

for i in range(len(audio_gcam_hists_standar)):
    axs[1].plot(audio_gcam_hists_counter[i,:],color="#ff9175", rasterized=True)
for i in range(len(audio_gcam_hists_counter)):
    axs[1].plot(audio_gcam_hists_standar[i,:],color="#4d9157", rasterized=True)
axs[1].plot(audio_func1, color="red", rasterized=True)
axs[1].plot(audio_func2, color="green", rasterized=True)
axs[1].set(xlabel="Length of packet (bytes)", ylabel="Relevance value of the byte in \nflow and mean of clusters (lines)", title="Audio packets representative functions vs all Audio histograms")

print("Processing Email")

for i in range(len(email_gcam_hists_standar)):
    axs[2].plot(email_gcam_hists_counter[i,:],color="#ff9175", rasterized=True)
for i in range(len(email_gcam_hists_counter)):
    axs[2].plot(email_gcam_hists_standar[i,:],color="#4d9157", rasterized=True)
axs[2].plot(email_func1, color="red", rasterized=True)
axs[2].plot(email_func2, color="green", rasterized=True)
axs[2].set(xlabel="Length of packet (bytes)", ylabel="Relevance value of the byte in \nflow and mean of clusters (lines)", title="Email packets representative functions vs all Email histograms")

print("Processing FTPS")

for i in range(len(ftps_gcam_hists_standar)):
    axs[3].plot(ftps_gcam_hists_counter[i,:],color="#ff9175", rasterized=True)
for i in range(len(ftps_gcam_hists_counter)):
    axs[3].plot(ftps_gcam_hists_standar[i,:],color="#4d9157", rasterized=True)
axs[3].plot(ftps_func1, color="red", rasterized=True)
axs[3].plot(ftps_func2, color="green", rasterized=True)
axs[3].set(xlabel="Length of packet (bytes)", ylabel="Relevance value of the byte in \nflow and mean of clusters (lines)", title="FTPS packets representative functions vs all FTPS histograms")

print("Processing Video")


for i in range(len(video_gcam_hists_standar)):
    axs[4].plot(video_gcam_hists_counter[i,:],color="#ff9175", rasterized=True)
for i in range(len(video_gcam_hists_counter)):
    axs[4].plot(video_gcam_hists_standar[i,:],color="#4d9157", rasterized=True)
axs[4].plot(video_func1, color="red", rasterized=True)
axs[4].plot(video_func2, color="green", rasterized=True)
axs[4].set(xlabel="Length of packet (bytes)", ylabel="Relevance value of the byte in \nflow and mean of clusters (lines)", title="Video packets representative functions vs all Video histograms")

print("Saving RepresentativeFunctions (can take a while)... ")
plt.savefig('GradCAM_RepresentativeFunctions.pdf',bbox_inches='tight')  

######################################################################
### 11 REPRESENTATIVE VS LENGTH HISTOGRAM
print("### 11 REPRESENTATIVE VS LENGTH HISTOGRAM")
plt.clf()

fig, axs = plt.subplots(5)
fig.set_size_inches(6.5, 18)
#fig.suptitle('GradCAM + Histogramas')
plt.subplots_adjust(left=0.2,
                    bottom=0, 
                    right=1, 
                    top=1, 
                    wspace=0.4, 
                    hspace=0.4)
axs[0].plot(chat_func2)
axs[0].plot(chat_func1)
axs[0].hist(chat_len_hist, 30)
axs[0].set(xlabel="Length of packet (bytes)", ylabel="Histogram of packets in category (green)\n and relevance of byte functions (lines)", title="Chat representative functions vs class histogram")
axs[0].legend(["Standard GradCAM","counterfactual GradCAM"])
axs[1].plot(audio_func1)
axs[1].plot(audio_func2)
axs[1].hist(audio_len_hist, 30)
axs[1].set(xlabel="Length of packet (bytes)", ylabel="Histogram of packets in category (green)\n and relevance of byte functions (lines)", title="Audio representative functions vs class histogram")
axs[1].legend(["Standard GradCAM","counterfactual GradCAM"])
axs[2].plot(email_func1)
axs[2].plot(email_func2)
axs[2].hist(email_len_hist, 30)
axs[2].set(xlabel="Length of packet (bytes)", ylabel="Histogram of packets in category (green)\n and relevance of byte functions (lines)", title="Email representative functions vs class histogram")
axs[2].legend(["Standard GradCAM","counterfactual GradCAM"])
axs[3].plot(ftps_func1)
axs[3].plot(ftps_func2)
axs[3].hist(ftps_len_hist, 30)
axs[3].set(xlabel="Length of packet (bytes)", ylabel="Histogram of packets in category (green)\n and relevance of byte functions (lines)", title="FTPS representative functions vs class histogram")
axs[3].legend(["Standard GradCAM","counterfactual GradCAM"],loc="upper left")
axs[4].plot(video_func1)
axs[4].plot(video_func2)
axs[4].hist(video_len_hist, 80)
axs[4].set(xlabel="Length of packet (bytes)", ylabel="Histogram of packets in category (green)\n and relevance of byte functions (lines)", title="Video representative functions vs class histogram")
axs[4].legend(["Standard GradCAM","counterfactual GradCAM"])
plt.savefig('GradCAM_Histograms_2.pdf',bbox_inches='tight')  
