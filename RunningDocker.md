# Running with docker
All commands should be run from the repository directory.

## Building images
First build the openvslam-socket image with
```bash
docker build -t openvslam-socket -f Dockerfile.socket . --build-arg NUM_THREADS=4
```
This builds the image with four threads. The NUM_THREADS argument can be 
used to use more or less if needed.

Then build the server image with 
```bash
cd viewer
docker build -t openvslam-server .
```

## Running the containers on Linux
First start the server:
```bash
docker run --rm --name openvslam-server --net=host openvslam-server
```
After starting the server, go to [http://localhost:3001/](http://localhost:3001/)
in a browser.

Then start openvslam-socket
```bash
docker run --rm -it --name openvslam-socket --net=host openvslam-socket
```
This will give you a shell to run openvslam example programs.
If you need access to other files in your computer from the openvslam shell,
you can add directories to the argument list of docker run before the last argument:
```bash
--volume /path/to/dataset/dir/:/dataset:ro
```
This will allow you to access the files from /dataset in the openvslam shell.
For example if both a dataset and a vocabulary file are needed:
```bash
# launch a container of openvslam-socket with --volume option
docker run --rm -it --name openvslam-socket --net=host \
    --volume /path/to/dataset/dir/:/dataset:ro \
    --volume /path/to/vocab/dir:/vocab:ro \
    openvslam-socket
# dataset/ and vocab/ are found at the root directory in the container
root@0c0c9f115d74:/# ls /
...   dataset/   vocab/   ...
```
The openvslam shell can be exited with the exit command.

For instuctions for macOs and more information see the OpenVSLAM 
documentation for the
[socket viewer](https://openvslam.readthedocs.io/en/master/docker.html#instructions-for-socketviewer).
