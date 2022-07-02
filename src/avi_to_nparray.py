import numpy as np
import os
import skvideo
skvideo.setFFmpegPath('D://Download_D/ffmpeg-2022-06-22-git-fed07efcde-essentials_build/ffmpeg-2022-06-22-git-fed07efcde-essentials_build/bin')

import skvideo.io

# assign directory
directory = '../src/YouTubeClips/'
count = 0
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    # checking if it is a file
    #if os.path.isfile(f):
    videodata = skvideo.io.vread(f)
    filename = filename.split('.')[0] + '.npy'
    print(filename, count)
    np.save(os.path.join('D:/video_annotation_data/MSVD/frames', filename), videodata) # Save to npy for quick loading later.
    count += 1

print(count)

