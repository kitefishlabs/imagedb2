# ImageDB2

Image indexing, DB, and recombination

## Dependencies/Setup

Linux Mint 19...

`apt-get install python3 python3-pip virtualenv`

python3 (aliased in bashrc as python if not already mapped)

pip already maps to pip3, to upgrade it: `pip install --upgrade --user pip`

Installed 'scipy stack](https://www.scipy.org/install.html): `python -m pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose`

Installed virtualenv: `apt-get install pyenv` and used pip to install pipenv (dep manager) `python3 -m pip install --user pipenv`

Now, inside of the imagedb2 folder, you can use pipenv to install deps: `pipenv install numpy`. The first run automatically creates and sets up a virtual environment for this project/directory. The proper files are automatically created in the `~/.local` dir. Each subsequent call to install via pipenv installs further deps into this specific virtual env. You will want to `pipenv install ipython` and then you'll have a shell via `pipenv shell`...

Install the following: ipython, numpy

### OpenCV 4.0

Follow the opencv [directions to build from source](https://docs.opencv.org/4.0.0/d2/de6/tutorial_py_setup_in_ubuntu.html)

These are the libs I installed on Mint (collected from other installation instructions and from cmake config output). Not all are needed, necesarrily.
* sudo apt-get install libpng-dev libjpeg-dev libtiff-dev libwebp-dev ffmpeg-dev 
* sudo apt-get install libgtk-3-dev
* libgstreamer-plugins-base1.0-dev
* sudo apt-get install libavcodec-dev libavutil-dev libswscale-dev libavformat-dev libavresample-dev
* sudo apt-get install libatlas-base-dev libopenblas-dev libtbb2 libtbb-dev

Also of note, java config was not automatically detected or it was not setup as OpenCV expected; did not build java or js bindings. Not sure if BLAS or LaPack support was actually included or relevant. No jasper; why is it even included as an optional dep?

Otherwise, things proceeded as expected. Additional links to ponder further installation/build details:
* https://towardsdatascience.com/how-to-install-opencv-and-extra-modules-from-source-using-cmake-and-then-set-it-up-in-your-pycharm-7e6ae25dbac5
* https://docs.opencv.org/4.0.1/da/df6/tutorial_py_table_of_contents_setup.html

## Sources/attribution for test images:

All images downloaded from [Wikimedia](https://commons.wikimedia.org)

Tiia Monto, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=51124937
Basile Morin - CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=70112005
Ester Inbar, Attribution, https://commons.wikimedia.org/w/index.php?curid=2429177
Moon 0903 - CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=21790551
Francisco Anzola - Gangnam, CC BY 2.0, https://commons.wikimedia.org/w/index.php?curid=32183429
Gzzz - CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=15920264
Weatherman1126, Public Domain, https://commons.wikimedia.org/w/index.php?curid=2993854
Beau Wade, CC BY 2.0, https://commons.wikimedia.org/w/index.php?curid=3432086

---
Copyright 2019 Tom Stoll (Kitefish Labs), distributed under the terms of the GNU General Public License, version 3.