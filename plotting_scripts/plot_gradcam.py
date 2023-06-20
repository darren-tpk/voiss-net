# Import all dependencies
import numpy as np
import glob
import cv2
import tensorflow as tf
from random import sample
from DataGenerator import DataGenerator
from keras.models import load_model
from matplotlib import pyplot as plt
import colorcet as cc

# Plotting options
station = 'PVV'
random_sample = False  # If True, randomly pick one file from each class from npy directory
visualize_standardized = True
gradcam_cmap = 'alpha'  # any matplotlib colormap, or 'alpha'

# Define GradCAM class
class GradCAM:
    """Code adopted from Pyimagesearch:
    URL https://pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    """

    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                     self.model.output])

        # record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the
            # associated with the specific class index
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel([inputs])
            best_prob = predictions[:, self.classIdx]
        # use automatic differentiation to compute the gradients
        grads = tape.gradient(best_prob, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castGrads * castConvOutputs * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        # normalize the heatmap such that all values lie in the range [0, 1]
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        return heatmap

# Define dictionary of spectrogram labels
label_dict = {0: 'Broadband Tremor',
              1: 'Harmonic Tremor',
              2: 'Monochromatic Tremor',
              3: 'Non-tremor Signal',
              4: 'Explosion',
              5: 'Noise'}

# Define test path dictionary for 6 examples (1 for each unique class)
if random_sample:
    test_paths = []
    npy_directory = '/Users/darrentpk/Desktop/all_npys/labeled_npy_4min/' + station + '*_'
    for label_index in list(label_dict.keys()):
        file_set = glob.glob(npy_directory + str(label_index) + '.npy')
        test_paths.append(sample(file_set,1)[0])
else:
    test_paths = ['/Users/darrentpk/Desktop/all_npys/labeled_npy_4min/' + station + '_202107270832_202107270836_0.npy',
                  '/Users/darrentpk/Desktop/all_npys/labeled_npy_4min/' + station + '_202107280236_202107280240_1.npy',
                  '/Users/darrentpk/Desktop/all_npys/labeled_npy_4min/' + station + '_202108050736_202108050740_2.npy',
                  '/Users/darrentpk/Desktop/all_npys/labeled_npy_4min/' + station + '_202108231824_202108231828_3.npy',
                  '/Users/darrentpk/Desktop/all_npys/labeled_npy_4min/' + station + '_202109210436_202109210440_4.npy',
                  '/Users/darrentpk/Desktop/all_npys/labeled_npy_4min/' + station + '_202109022056_202109022100_5.npy']
test_labels = [int(i.split("_")[-1][0]) for i in test_paths]
test_label_dict = dict(zip(test_paths, test_labels))

# Load the first filepath to get spectrogram shape
eg_spec = np.load(test_paths[0])
test_params = {
    "dim": eg_spec.shape,
    "batch_size": len(test_paths),
    "n_classes": len(label_dict),
    "shuffle": False,
}

# Load model
saved_model = load_model('/Users/darrentpk/Desktop/GitHub/tremor_ml/models/4min_all_subsampled2_model.h5')
saved_meanvar = np.load('/Users/darrentpk/Desktop/GitHub/tremor_ml/models/4min_all_subsampled2_meanvar.npy')
running_x_mean = saved_meanvar[0]
running_x_var = saved_meanvar[1]

# Initialize data generator for test
test_gen = DataGenerator(test_paths, test_label_dict, **test_params, is_training=False,
                         running_x_mean=running_x_mean, running_x_var=running_x_var)

# Run model to get class-specific probabilities and best class
test = saved_model.predict(test_gen)
pred_labs = np.argmax(test, axis=1)
true_labs = np.array([test_gen.labels[id] for id in test_gen.list_ids])

# Print predictions
print('True labels:', true_labs)
print('Predicted  :', pred_labs)

### Now plot GradCAM

# Initialize multiplot
fig, axs = plt.subplots(3, 4, figsize=(8, 7.5))

# Loop over the 6 test cases
for index in range(len(test_paths)):

    # Load spectrogram slice
    raw_spec = np.load(test_paths[index])

    # Process spectrogram slice by standardizing and min-max scaling
    spec = (raw_spec - running_x_mean) / np.sqrt(running_x_var + 1e-5)
    spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))

    # Extract gradients and normalize
    gc = GradCAM(saved_model, index)
    spec_tensor = tf.reshape(spec, shape=[-1, *np.shape(spec), 1])
    heatmap = gc.compute_heatmap(spec_tensor)
    heatmap /= np.max(heatmap)

    # Plot base spectrogram
    if visualize_standardized:
        axs.ravel()[2*index].imshow(spec, cmap=cc.cm.rainbow, origin='lower', aspect='auto', interpolation='None',
                                  vmin=np.percentile(spec, 20), vmax=np.percentile(spec, 97.5))
    else:
        axs.ravel()[2*index].imshow(raw_spec, cmap=cc.cm.rainbow, origin='lower', aspect='auto', interpolation='None',
                                  vmin=np.percentile(raw_spec, 20), vmax=np.percentile(raw_spec, 97.5))

    # Plot spectrogram with gradcam overlay
    if gradcam_cmap != 'alpha' and visualize_standardized:
        axs.ravel()[2*index+1].imshow(spec, cmap='gist_yarg', origin='lower', aspect='auto', interpolation='None',
                                  vmin=np.percentile(spec, 20), vmax=np.percentile(spec, 97.5))
        axs.ravel()[2 * index + 1].imshow(heatmap, cmap=gradcam_cmap, origin='lower', aspect='auto', interpolation='None', alpha=0.45)
    elif gradcam_cmap != 'alpha' and not visualize_standardized:
        axs.ravel()[2*index+1].imshow(raw_spec, cmap='gist_yarg', origin='lower', aspect='auto', interpolation='None',
                                  vmin=np.percentile(raw_spec, 20), vmax=np.percentile(raw_spec, 97.5))
        axs.ravel()[2 * index + 1].imshow(heatmap, cmap=gradcam_cmap, origin='lower', aspect='auto', interpolation='None', alpha=0.45)
    elif gradcam_cmap == 'alpha' and visualize_standardized:
        axs.ravel()[2*index+1].imshow(spec, cmap=cc.cm.rainbow, origin='lower', aspect='auto', interpolation='None', alpha=heatmap**2,
                                  vmin=np.percentile(spec, 20), vmax=np.percentile(spec, 97.5))
    elif gradcam_cmap == 'alpha' and not visualize_standardized:
        axs.ravel()[2 * index + 1].imshow(raw_spec, cmap=cc.cm.rainbow, origin='lower', aspect='auto', alpha=heatmap**2,
                                          interpolation='None', vmin=np.percentile(raw_spec, 20), vmax=np.percentile(raw_spec, 97.5))
    else:
        raise ValueError('Invalid GradCAM colormap! ')


    # Fix spectrogram axes and ticks
    axs.ravel()[2*index].set_title('%s (%d)' % (label_dict[index],true_labs[index]), fontsize=10)
    axs.ravel()[2*index].set_xticks([])
    axs.ravel()[2*index].set_yticks([])

    # Fix gradcam axes and ticks
    if true_labs[index] == pred_labs[index]:
        axs.ravel()[2*index+1].set_title('GradCAM (%d)' % pred_labs[index],
                                         fontsize=10, color='darkgreen', fontweight='bold')
        for spine in axs.ravel()[2*index+1].spines.values():
            spine.set_edgecolor('darkgreen')
            spine.set_linewidth(3)
    else:
        axs.ravel()[2*index+1].set_title('GradCAM (%d)' % pred_labs[index],
                                         fontsize=10, color='darkred', fontweight='bold')
        for spine in axs.ravel()[2*index+1].spines.values():
            spine.set_edgecolor('darkred')
            spine.set_linewidth(3)
    axs.ravel()[2*index+1].set_xticks([])
    axs.ravel()[2*index+1].set_yticks([])

# Show figure
fig.suptitle('GradCAM Examples for Station %s' % station, fontsize=13, fontweight='bold')
fig.show()
