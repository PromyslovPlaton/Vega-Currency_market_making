# Do you using Colab?
try:
    from google.colab import drive
    %tensorflow_version 2.x
    COLAB = True
    print("Note: using Google CoLab")
except:
    print("Note: not using Google CoLab")
    COLAB = False

if COLAB:
    !sudo apt-get install xvfb ffmpeg x11-utils
    !pip install 'gym==0.17.3'
    !pip install 'imageio==2.4.0'
    !pip install PILLOW
    !pip install 'pyglet==1.3.2'
    !pip install pyvirtualdisplay
    !pip install 'tf-agents==0.12.0'
    !pip install imageio-ffmpeg
    print("Note: done for Colab!")
else:
    print("Note: done for PC!")