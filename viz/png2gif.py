import imageio
import os

folder_path = r''
save_path = r'./Driving'

if not os.path.exists(save_path):
    os.makedirs(save_path)
# save_path = os.path.join(save_path, os.path.basename(folder_path) + '.gif')

fps = 30  # specify the fps of the gif

# get the list of png files in the folder
file_list = os.listdir(folder_path)
png_list = [f for f in file_list if f.endswith('.png')]

# sort the png files in ascending order
png_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

# read the png files and create the gif
with imageio.get_writer(save_path, mode='I', fps=fps) as writer:
    for png_file in png_list:
        image = imageio.imread(os.path.join(folder_path, png_file))
        writer.append_data(image)
