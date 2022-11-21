"""
    Nama : Dhiya Fakhar Nafi
    Kelas : D4 TI-3A
    NIM : 201524002

    Colorization based on the Zhang Image Colorization Deep Learning Algorithm

    Algoritma yang digunakan berasal dari tautan berikut:
    https://www.pyimagesearch.com/2019/02/25/black-and-white-image-colorization-with-opencv-and-deep-learning/

    Data untuk melakukan training sebagai berikut:
    https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1

    Library PySimpleGUI digunakan untuk membuat antarmuka.
"""

# Menggunakan Library NumPy dan OpenCV untuk mengolah citra
import numpy as np
import cv2

# Menggunakan Library Image dan I/O (Input-Output) untuk membuka citra
from PIL import Image
import io

# Menggunakan PySimpleGUI untuk membuat GUI
import PySimpleGUI as sg
import os.path

# Versi dari program
version = '21 November 2022'

"""
BAGIAN DIBAWAH INI JANGAN DIUBAH!
"""

# Membaca file prototxt, models, dan point
prototxt = r'model/colorization_deploy_v2.prototxt'
model = r'model/colorization_release_v2.caffemodel'
points = r'model/pts_in_hull.npy'

# Membuka file-file yang dibutuhkan
points = os.path.join(os.path.dirname(__file__), points)
prototxt = os.path.join(os.path.dirname(__file__), prototxt)
model = os.path.join(os.path.dirname(__file__), model)

# Jika file tidak ditemukan, tampilkan pesan error
if not os.path.isfile(model):
    sg.popup_scrolled('Model file tidak ditemukan!', 'File yang anda butuhkan adalah "colorization_release_v2.caffemodel"',
                      'Unduh dan letakkan pada folder model', 'Unduh disini:\n', r'https://www.dropbox.com/s/dx0qvhhp5hbcx7z/colorization_release_v2.caffemodel?dl=1')
    exit()

"""
BAGIAN DIATAS JANGAN DIUBAH!
"""

"""
Let’s go ahead and use argparse to parse command line arguments. This script requires that these four arguments be passed to the script directly from the terminal:

    --image
    : The path to our input black/white image.
    --prototxt
    : Our path to the Caffe prototxt file.
    --model
    . Our path to the Caffe pre-trained model.
    --points
    : The path to a NumPy cluster center points file.

With the above four flags and corresponding arguments, the script will be able to run with different inputs without changing any code.
"""

# Baca file model yang akan menjadi bahan training AI
net = cv2.dnn.readNetFromCaffe(prototxt, model) # Loads our Caffe model directly from the command line argument values. OpenCV can read Caffe models via the cv2.dnn.readNetFromCaffe function.
pts = np.load(points) # Loads the cluster center points directly from the command line argument path to the points file. This file is in NumPy format so we’re using np.load.

# Add the cluster centers as 1x1 convolutions to the model
# Load centers for ab channel quantization used for rebalancing.
# Treat each of the points as 1×1 convolutions and add them to the model.
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh") 
pts = pts.transpose().reshape(2, 313, 1, 1) 
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Fungsi untuk mewarnai citra
# Parameter input : Nama file dan cv2 frame
def colorize_image(image_filename=None, cv2_frame=None):
    
    """
    Penjelasan:
    1. Parameter image_filename: nama file
    2. Parameter cv2_frame: (cv2 frame)
    
    Output/Keluaran: Citra sebelum dan sesudah diwarnai
    """

    # 1. Muat alamat folder citra, ubah skala intensitas piksel ke kisaran [0, 1], lalu konversikan gambar dari BGR ke ruang warna (Channel) Lab
    image = cv2.imread(image_filename) if image_filename else cv2_frame
    scaled = image.astype("float32") / 255.0 # Scaling pixel intensities to the range [0, 1]
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB) # Converting from BGR to Lab color space 

    # 2. Ubah ukuran citra menjadi 224x224 (dimensi yang diterima dalam Color Network), pisahkan channel L,a, dan b, ekstrasi nilai L, lalu lakukan pemusatan rata-rata pada channel L (50)
    resized = cv2.resize(lab, (224, 224)) # the required input dimensions for the network.
    L = cv2.split(resized)[0] 
    L -= 50 # (Cahaya citra dikurangi sebesar 50)

    # 3. Gunakan Channel Warna L melalui jaringan yang akan memprediksi nilai Channel 'a' dan 'b'
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # 4. Ubah bentul 'ab' yang diprediksi ke dimensi yang sama dengan citra yang akan diwarnai
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    # 5. Ambil nilai L pada citra aslinya (bukan yang diperkecil) dan gabungkan Channel Warna 'L' asli dengan saluran 'ab' yang diprediksi
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # 6. Konversikan citra yang diwarnai dari Channel warna Lab ke RGB, lalu beri batas pada nilai rentang [0,1]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # 7. Gambar berwarna saat ini direpresentasikan sebagai tipe data floating point dalam rentang [0, 1] 
    # Kita ubah ke representasi bilangan bulat 8-bit dalam rentang [0, 255]
    colorized = (255 * colorized).astype("uint8")
    return image, colorized

# -------------- ANTARMUKA -----------------
# Aplikasi terbagi menjadi dua bagian

# Ikon Aplikasi
logo_image = Image.open('logo.png')
logo_png = io.BytesIO()
logo_image.save(logo_png, format="PNG")
logo_display = logo_png.getvalue()

# Bagian Kiri
left_col = [[sg.Image(data=logo_display)], 
            [sg.Text('Folder yang dipilih', font='Arial'), sg.In(size=(20,1), enable_events=True ,key='-FOLDER-'), sg.FolderBrowse()],
            [sg.Listbox(values=[], enable_events=True, size=(40,20),key='-FILE LIST-')],]

# Bagian Kanan
images_col = [[sg.In(enable_events=True, key='-IN FILE-')], # Abaikan
              [sg.Button('Simpan', key='-SAVE-'), sg.Button('Keluar')],
              [sg.Image(filename='', key='-IN-'), sg.Image(filename='', key='-OUT-')], # Menampilkan Citra Sebelum dan Sesudah Diwarnai.
              [sg.Text('Versi ' + version, font='Arial')],]

# ----- LAYOUT PENUH -----
layout = [[sg.Column(left_col), sg.Column(images_col)]]

# ----- MEMBUAT WINDOW APLIKASI -----
window = sg.Window('Perwarna', layout, grab_anywhere=True)

# ----- MULAI APLIKASI -----
prev_filename = colorized = cap = None
while True:
    event, values = window.read()
    
    # Jika mengklik tombol keluar
    if event in (None, 'Keluar'):
        break
    
    # Menampilkan kumpulan citra pada folder
    if event == '-FOLDER-':         
        folder = values['-FOLDER-']
        
        # Tipe citra harus PNG, JPG, JPEG, TIFF, dan BMP
        img_types = (".png", ".jpg", "jpeg", ".tiff", ".bmp")

        # Mendapatkan kumpulan citra pada folder
        try:
            flist0 = os.listdir(folder)
        except:
            continue
        fnames = [f for f in flist0 if os.path.isfile(
            os.path.join(folder, f)) and f.lower().endswith(img_types)]
        window['-FILE LIST-'].update(fnames)
    
    # Memilih Citra
    elif event == '-FILE LIST-':    # Jika file yang dipilih berasal dari suatu folder
        try:
            filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
            image = cv2.imread(filename)
            window['-IN-'].update(data=cv2.imencode('.png', image)[1].tobytes()) # Dalam bentuk PNG
            window['-OUT-'].update(data='')
            window['-IN FILE-'].update('')
            image, colorized = colorize_image(filename)

            window['-OUT-'].update(data=cv2.imencode('.png', colorized)[1].tobytes()) # Dalam bentuk PNG
        except:
            continue
    
    # Mewarnai dan Menampilkan Citra
    elif event == '-PHOTO-':        # Ketika tombol "Warnai" diklik
        try:
            # Jika Citra yang dipilih berasal dari alamat folder yang dipilih
            if values['-IN FILE-']:
                filename = values['-IN FILE-']  # Citra pada alamat yang akan dipilih
            # Jika Citra yang dipilih berasal dari kumpulan file pada folder yang dipilih
            elif values['-FILE LIST-']:
                filename = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0]) # Citra pada folder yang akan dipilih

            else:
                continue
                image, colorized = colorize_image(filename) # Mewarnai citra

                # Perbarui Citra
                window['-IN-'].update(data=cv2.imencode('.png', image)[1].tobytes())
            
            # Tampilkan citra
            window['-OUT-'].update(data=cv2.imencode('.png', colorized)[1].tobytes())
        except:
            continue
    
    # Jika satu file
    elif event == '-IN FILE-':      # Jika suatu file dipilih
        filename = values['-IN FILE-']
        
        # Jika filenya tidak sama
        if filename != prev_filename:
            prev_filename = filename
            try:
                image = cv2.imread(filename)

                # Inputan yang diambil berasal dari file png
                window['-IN-'].update(data=cv2.imencode('.png', image)[1].tobytes())
            except:
                continue
    
    # Simpan Citra
    elif event == '-SAVE-' and colorized is not None:   # Clicked the Save File button
        filename = sg.popup_get_file('Simpan Citra.', save_as=True)
        try:
            if filename:
                # Membuat output citra berwarna.
                cv2.imwrite(filename, colorized)
                
                # JIka berhasil, tampilkan pop-up.
                sg.popup_quick_message('Simpan Citra Berhasil!', background_color='red', text_color='white', font='Any 16')
        except:
            # JIka gagal, tampilkan pop-up.
            sg.popup_quick_message('Simpan Citra Gagal!', background_color='red', text_color='white', font='Any 16')

# ----- MENUTUP PROGRAM -----
window.close()