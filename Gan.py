import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

numHiddenLayerNodes = 192
batch_size = 100
gen_noise_size = 98

def xavierInit(size):
    return tf.random_normal(shape=size, stddev=tf.sqrt(2. / (size[0] + size[1])))

def activationFunc(features):
    return tf.nn.relu(features)

D_in = tf.placeholder(tf.float32, shape=[None, 784])
def DNetInit(numNodes):
    W1 = tf.Variable(xavierInit([784, numNodes]))
    b1 = tf.Variable(tf.constant(0.1, shape=[numNodes]))
    W2 = tf.Variable(xavierInit([numNodes, 1]))
    b2 = tf.Variable(tf.constant(0.1, shape=[1]))
    return [W1, b1, W2, b2]

DLayers = DNetInit(numHiddenLayerNodes)

G_in = tf.placeholder(tf.float32, shape=[None, gen_noise_size])
def GNetInit(numNodes):
    W1 = tf.Variable(xavierInit([gen_noise_size, numNodes]))
    b1 = tf.Variable(tf.constant(0.1, shape=[numNodes]))
    W2 = tf.Variable(xavierInit([numNodes, 784]))
    b2 = tf.Variable(tf.constant(0.1, shape=[784]))
    return [W1, b1, W2, b2]

GLayers = GNetInit(numHiddenLayerNodes)

def generator(g_in):
    hidden = tf.nn.dropout(activationFunc(tf.matmul(g_in, GLayers[0]) + GLayers[1]), .75)
    out = tf.nn.sigmoid(tf.matmul(hidden, GLayers[2]) + GLayers[3])
    return out


def discriminator(d_in):
    hidden = tf.nn.dropout(activationFunc(tf.matmul(d_in, DLayers[0]) + DLayers[1]), .75)
    out_logit = tf.matmul(hidden, DLayers[2]) + DLayers[3]
    out = tf.nn.sigmoid(out_logit)
    return out, out_logit

def plot(samples):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


Gen_image = generator(G_in)
D_prob, D_logit = discriminator(D_in)
D_gen_prob, D_gen_logit = discriminator(Gen_image)

D_cost = -tf.reduce_mean(tf.log(D_prob) + tf.log(1. - D_gen_prob))
G_cost = -tf.reduce_mean(tf.log(D_gen_prob))

D_trainer = tf.train.AdamOptimizer().minimize(D_cost, var_list=DLayers)
G_trainer = tf.train.AdamOptimizer().minimize(G_cost, var_list=GLayers)



with tf.Session() as sess:
    minst_input_data = input_data.read_data_sets('../../minst_input_data_data', one_hot=True)

    sess.run(tf.global_variables_initializer())

    if not os.path.exists('out/'):
        os.makedirs('out/')

    i = 0

    for iteration in range(20001):
        if iteration % 2000 == 0:
            samples = sess.run(Gen_image, feed_dict={G_in: np.random.uniform(-1., 1., size=[25, gen_noise_size])})
            fig = plot(samples)
            plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            i += 1
            plt.close(fig)

        input_batch, _ = minst_input_data.train.next_batch(batch_size)

        _, D_confidence = sess.run([D_trainer, D_gen_prob], feed_dict={D_in: input_batch, G_in: np.random.uniform(-1., 1., size=[batch_size, gen_noise_size])})
        _, G_cost_curr = sess.run([G_trainer, G_cost], feed_dict={G_in: np.random.uniform(-1., 1., size=[batch_size, gen_noise_size])})

        if iteration % 2000 == 0:
            print('Iteration: {}'.format(iteration))
            print('D gen cost high: {:.4}'.format(np.max(D_confidence)))
            print('D gen cost avg: {:.4}'.format(np.mean(D_confidence)))
            print('G cost: {:.4}'.format(G_cost_curr))
            print()