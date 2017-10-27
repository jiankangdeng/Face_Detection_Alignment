import tensorflow as tf
import time
import networks
import traceback

slim = tf.contrib.slim

if __name__ == '__main__':
    while True:
        try:
            networks.DNFaceMultiView().train()
        except Exception as e:
            traceback.print_exc()
            time.sleep(10)
