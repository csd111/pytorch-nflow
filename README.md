# pytorch-nflow

A pytorch implementation of Normalizing Flow networks

This program is distributed in the hope that it will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose. See the GNU General Public License for more details.

## Getting Started

### Prerequisites
This project runs on Python 3. All the necessary Python packages are listed in ```requirements.txt```.

### Unit tests
You can run a bunch of unit tests from the command line.

```buildoutcfg
python -m unittest discover tests/
```

## Basic Usage

To start things off, you are advised to have a look at the training script ```train_model.py```.
In essence, you should be able to get things running just by changing the path to the root folder of your database of choice.

Please note that it requires the data to be availble as raw images in a pillow-readable format, separated into a "train" and a "val" subfolder.


## References

Tensorflow code: 
- Real-NVP https://github.com/tensorflow/models/tree/master/research/real_nvp
- OpenAI Glow https://github.com/openai/glow
- Flow++ https://github.com/aravindsrinivas/flowpp

Other pytorch implementations of normalizing flows :
- https://github.com/chrischute/glow
- https://github.com/karpathy/pytorch-normalizing-flows
- https://github.com/tonyduan/normalizing-flows
