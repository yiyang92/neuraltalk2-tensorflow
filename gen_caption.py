import tensorflow as tf
import numpy as np
import argparse
import os
import pickle

from utils.parameters import Parameters
from model.decoder import Decoder
from utils.captions import Dictionary
from utils.image_embeddings import vgg16
from utils.image_utils import load_image
# debuging
from tensorflow.python.tools import inspect_checkpoint as chkp

class Generator():
    """Generate caption, given the image
    """
    def __init__(self,
                 checkpoint_path,
                 params_path, vocab_path, gen_method='greedy'):
        self.checkpoint_path = checkpoint_path
        self.params = self._load_params(params_path)
        self.gen_method = gen_method
        # load vocabulary
        try:
            os.path.exists(vocab_path)
        except:
            raise ValueError("No caption vocabulary path specified, "
                       "Usually it can be found in the ./pickles foulder "
                       "after model training")
        with open(vocab_path, 'rb') as rf:
            data_dict = pickle.load(rf)
        self.data_dict = Dictionary(data_dict, self.params.keep_words)
        self.params.vocab_size =self.data_dict.vocab_size

    def _c_v_generator(self, image):
        # TODO: finish cluster vector implementation
        return None

    def _load_params(self, params_path):
        """Load serialized Parameters class, for convenience
        """
        with open(params_path, 'rb') as rf:
            params = pickle.load(rf)
        return params

    def generate_caption(self, img_path, beam_size=2):
        """Caption generator
        Args:
            image_path: path to the image
        Returns:
            caption: caption, generated for a given image
        """
        # TODO: to avoid specify model again use frozen graph
        g = tf.Graph()
        # change some Parameters
        self.params.sample_gen = self.gen_method
        try:
            os.path.exists(img_path)
        except:
            raise ValueError("Image not found")
        with g.as_default():
            # specify rnn_placeholders
            ann_lengths_ps = tf.placeholder(tf.int32, [None])
            captions_ps = tf.placeholder(tf.int32, [None, None])
            images_ps = tf.placeholder(tf.float32, [None, 224, 224, 3])
            with tf.variable_scope("cnn"):
                image_embeddings = vgg16(images_ps, trainable_fe=True,
                                         trainable_top=True)
            features = image_embeddings.fc2
            # image fesatures [b_size + f_size(4096)] -> [b_size + embed_size]
            images_fv = tf.layers.dense(features, self.params.embed_size,
                                        name='imf_emb')
            # will use methods from Decoder class
            decoder = Decoder(images_fv, captions_ps,
                              ann_lengths_ps, self.params, self.data_dict)
            with tf.variable_scope("decoder"):
                _, _ = decoder.decoder()  # for initialization
            # if use cluster vectors
            if self.params.use_c_v:
                # cluster vectors from "Diverse and Accurate Image Description.." paper.
                c_i = tf.placeholder(tf.float32, [None, 90])
                c_i_emb = tf.layers.dense(c_i, self.params.embed_size,
                                          name='cv_emb')
                # map cluster vectors into embedding space
                decoder.c_i = c_i_emb
                decoder.c_i_ph = c_i
            # image_id
            im_id = [img_path.split('/')[-1]]
            saver = tf.train.Saver(tf.trainable_variables())
        with tf.Session(graph=g) as sess:
            saver.restore(sess, self.checkpoint_path)
            if self.params.use_c_v:
                c_v = self._c_v_generator(image)
            else:
                c_v = None
            im_shape = (224, 224) # VGG16
            image = np.expand_dims(load_image(img_path, im_shape), 0)
            if self.gen_method == 'beam_search':
                sent = decoder.beam_search(sess, im_id, image,
                                           images_ps, c_v,
                                           beam_size=beam_size)
            elif self.gen_method == 'greedy':
                sent, _ = decoder.online_inference(sess, im_id, image,
                                                   images_ps, c_v=c_v)
            return sent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify generation parameters")
    parser.add_argument('--img_path', required=True,
                        help="Path to the image")
    parser.add_argument('--checkpoint', required=True,
                        help="Model checkpoint path")
    parser.add_argument('--vocab_path', default='./pickles/capt_vocab.pickle',
                        help="Indices to words dictionary")
    parser.add_argument('--gpu', default='',
                        help="Specify GPU number if use GPU")
    parser.add_argument('--c_v_generator', default=None,
                        help="If use cluster vectors, specify tensorflow api model"
                        "For more information look README")
    parser.add_argument('--gen_method', default='greedy',
                        help='greedy, beam_search or sample')
    parser.add_argument('--params_path', required=True, default=None,
                        help="specify params pickle file")
    parser.add_argument('--beam_size', default=2,
                        help="If using beam_search, specify beam_size")
    args = parser.parse_args()
    # CUDA settings
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    # parameter of the model
    params = Parameters()
    generator = Generator(checkpoint_path=args.checkpoint,
                          params_path=args.params_path,
                          vocab_path=args.vocab_path,
                          gen_method=args.gen_method)
    caption = generator.generate_caption(args.img_path, args.beam_size)
    print(caption[0]['caption'])
