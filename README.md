Tweaked https://github.com/jcjohnson/torch-rnn to implement line-by-line based chatbot. Since I made too many changes, it's not compatible with the original, hence I created a separate repository.  In comparison to the original torch-rnn

####removed:

* creation and using validation and test sets
* modelreset and modelclear before saving the model to disk

####replaced:

* char by char (`..`) concatenation during sampling with table concatenation
* "xxx byte limitation" sampling with "until new line" one
* creating the .h5 conversion table based on the current .txt file with a universal one (ASCII only)


####added:

* GRU cells from https://github.com/guillitte/torch-rnn
* GridGRU (or GridLSTM) from https://github.com/guillitte/torch-rnn/tree/Dev
* Server implementation from https://github.com/jcjohnson/torch-rnn/pull/61 (requires https://github.com/kernelsauce/turbo)
* Dia object, that seeds the NN with the last "n" conversation lines
* Interactive dialogue
* Command structure to change parameters "on the fly"
* Reset program to reset the saved models in order to either train them with a different batch/wondow size or for sampling
* Timer (to show time per training iteration)

##Examples:

```txt
th mysample.lua -temperature 0.7 -checkpoint cp_3x1000-0.1-1_7.8125e-06_14760.t7.reset.t7
Loading 	cp_3x1000-0.1-1_7.8125e-06_14760.t7.reset.t7	
Loaded
Me: What is your name?
NN: My name is Lancelot.
Me: Nice name. And with a history. Or... are you THAT Lancelot?
NN: Yes, my lady.
Me: Wow...
. . .
Me: ;curlen
The current length is 5, the maximum length is 16
Me: ;setlen=5
The dialogue length is set to 5
Me: ;temp=0.6
The temp is set to 0.6
Me: ;reset
Reset
Me: Hello
NN: I'm asking you to leave.
Me: Why?
NN: Well, I'm not a woman.
Me: Nobody's perfect.
```

Original Readme from https://github.com/jcjohnson/torch-rnn
-------------
# torch-rnn
torch-rnn provides high-performance, reusable RNN and LSTM modules for torch7, and uses these modules for character-level
language modeling similar to [char-rnn](https://github.com/karpathy/char-rnn).

You can find documentation for the RNN and LSTM modules [here](doc/modules.md); they have no dependencies other than `torch`
and `nn`, so they should be easy to integrate into existing projects.

Compared to char-rnn, torch-rnn is up to **1.9x faster** and uses up to **7x less memory**. For more details see 
the [Benchmark](#benchmarks) section below.


# Installation

## Docker Images
Cristian Baldi has prepared Docker images for both CPU-only mode and GPU mode;
you can [find them here](https://github.com/crisbal/docker-torch-rnn).

## System setup
You'll need to install the header files for Python 2.7 and the HDF5 library. On Ubuntu you should be able to install
like this:

```bash
sudo apt-get -y install python2.7-dev
sudo apt-get install libhdf5-dev
```

## Python setup
The preprocessing script is written in Python 2.7; its dependencies are in the file `requirements.txt`.
You can install these dependencies in a virtual environment like this:

```bash
virtualenv .env                  # Create the virtual environment
source .env/bin/activate         # Activate the virtual environment
pip install -r requirements.txt  # Install Python dependencies
# Work for a while ...
deactivate                       # Exit the virtual environment
```

## Lua setup
The main modeling code is written in Lua using [torch](http://torch.ch); you can find installation instructions
[here](http://torch.ch/docs/getting-started.html#_). You'll need the following Lua packages:

- [torch/torch7](https://github.com/torch/torch7)
- [torch/nn](https://github.com/torch/nn)
- [torch/optim](https://github.com/torch/optim)
- [lua-cjson](https://luarocks.org/modules/luarocks/lua-cjson)
- [torch-hdf5](https://github.com/deepmind/torch-hdf5)

After installing torch, you can install / update these packages by running the following:

```bash
# Install most things using luarocks
luarocks install torch
luarocks install nn
luarocks install optim
luarocks install lua-cjson

# We need to install torch-hdf5 from GitHub
git clone https://github.com/anibali/torch-hdf5.git
cd torch-hdf5
git checkout hdf5-1.10 
luarocks make hdf5-0-0.rockspec
```

### CUDA support (Optional)
To enable GPU acceleration with CUDA, you'll need to install CUDA 6.5 or higher and the following Lua packages:
- [torch/cutorch](https://github.com/torch/cutorch)
- [torch/cunn](https://github.com/torch/cunn)

You can install / update them by running:

```bash
luarocks install cutorch
luarocks install cunn
```

## OpenCL support (Optional)
To enable GPU acceleration with OpenCL, you'll need to install the following Lua packages:
- [cltorch](https://github.com/hughperkins/cltorch)
- [clnn](https://github.com/hughperkins/clnn)

You can install / update them by running:

```bash
luarocks install cltorch
luarocks install clnn
```

## OSX Installation
Jeff Thompson has written a very detailed installation guide for OSX that you [can find here](http://www.jeffreythompson.org/blog/2016/03/25/torch-rnn-mac-install/).

# Usage
To train a model and use it to generate new text, you'll need to follow three simple steps:

## Step 1: Preprocess the data
You can use any text file for training models. Before training, you'll need to preprocess the data using the script
`scripts/preprocess.py`; this will generate an HDF5 file and JSON file containing a preprocessed version of the data.

If you have training data stored in `my_data.txt`, you can run the script like this:

```bash
python scripts/preprocess.py \
  --input_txt my_data.txt \
  --output_h5 my_data.h5 \
  --output_json my_data.json
```

This will produce files `my_data.h5` and `my_data.json` that will be passed to the training script.

There are a few more flags you can use to configure preprocessing; [read about them here](doc/flags.md#preprocessing)

## Step 2: Train the model
After preprocessing the data, you'll need to train the model using the `train.lua` script. This will be the slowest step.
You can run the training script like this:

```bash
th train.lua -input_h5 my_data.h5 -input_json my_data.json
```

This will read the data stored in `my_data.h5` and `my_data.json`, run for a while, and save checkpoints to files with 
names like `cv/checkpoint_1000.t7`.

You can change the RNN model type, hidden state size, and number of RNN layers like this:

```bash
th train.lua -input_h5 my_data.h5 -input_json my_data.json -model_type rnn -num_layers 3 -rnn_size 256
```

By default this will run in GPU mode using CUDA; to run in CPU-only mode, add the flag `-gpu -1`.

To run with OpenCL, add the flag `-gpu_backend opencl`.

There are many more flags you can use to configure training; [read about them here](doc/flags.md#training).

## Step 3: Sample from the model
After training a model, you can generate new text by sampling from it using the script `sample.lua`. Run it like this:

```bash
th sample.lua -checkpoint cv/checkpoint_10000.t7 -length 2000
```

This will load the trained checkpoint `cv/checkpoint_10000.t7` from the previous step, sample 2000 characters from it,
and print the results to the console.

By default the sampling script will run in GPU mode using CUDA; to run in CPU-only mode add the flag `-gpu -1` and
to run in OpenCL mode add the flag `-gpu_backend opencl`.

There are more flags you can use to configure sampling; [read about them here](doc/flags.md#sampling).

## Step 4 (optional): Serve samples from an HTTP server
If you want to generate new text on demand without loading the model for every sample, you can use the script `server.lua`. 
You will need to install [Turbo](https://github.com/kernelsauce/turbo) framework first:
```bash
luarocks install turbo
```

Check [Turbo installation manual](https://github.com/kernelsauce/turbo#installation) in case of problems.
Then run the server:

```bash
th server.lua -checkpoint cv/checkpoint_10000.t7 -port 8888
```

Now you can generate new sample by sending an HTTP GET request:
```bash
curl -G -d "length=100&temperature=0.9" http://localhost:8888/sample
```

The following command line arguments of `sample.lua` remain unchanged for `server.lua`: 
`-checkpoint`, `-gpu`, `-gpu_backend`, `-verbose`. `-port` argument configures the HTTP port.
The other arguments (`length`, `start_text`, `temperature` and `sample`) should be passed as GET parameters.



