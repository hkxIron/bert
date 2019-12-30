from modeling import *
import numpy as np
np.random.seed(0)
sess = tf.Session()

def f1():
    batch_size=2
    seq_length=3

    input_ids=tf.constant(np.random.randint(0, 100, size=(batch_size, seq_length)))
    input_mask=tf.constant(np.random.randint(2, size=(batch_size, seq_length)))
    attention_mask = create_attention_mask_from_input_mask(input_ids, input_mask)

    print("input ids:\n", sess.run(input_ids))
    print("input mask:\n", sess.run(input_mask))
    print("attention mask:\n", sess.run(attention_mask))
    """
    input ids:
     [[44 47 64]
     [67 67  9]]
    input mask:
     [[1 1 0]
     [0 1 0]]
    attention mask:
     [[[1. 1. 0.]
      [1. 1. 0.]
      [1. 1. 0.]]

     [[0. 1. 0.]
      [0. 1. 0.]
      [0. 1. 0.]]]
    """

f1()

sess.close()
