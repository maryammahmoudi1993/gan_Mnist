# Data

This tutorial uses the **MNIST database of handwritten digits**: 70,000 grey-scale pictures of digits, 28 x 28 pixels each
(60,000 for training, 10,000 for testing). The GAN only needs the 60,000 training pictures; the labels are not used.

- Copyright: Yann LeCun and Corinna Cortes. MNIST is a derivative work of the original NIST data sets and is made available under the
  [Creative Commons Attribution-Share Alike 3.0 licence](https://creativecommons.org/licenses/by-sa/3.0/).
  (This statement is taken from the documentation of the data set in the Keras package used here, version 3.15.1; I could not open the
  authors' website from the environment in which this tutorial was written.)
- Homepage: <http://yann.lecun.com/exdb/mnist/>.

You do not download anything by hand. The first time the code runs, Keras stores the file in `~/.keras/datasets/` (about 11 MB).
Nothing is stored inside this repository.
