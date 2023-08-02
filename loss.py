import tensorflow as tf

def loss_matching(d1,d2,margin):
    #return tf.sqrt(tf.reduce_sum(tf.square(d1 - d2), axis=1)+1e-7)
    pair_dist = tf.sqrt(tf.reduce_sum(tf.square(d1 - d2), axis=1)+1e-7)
    return tf.nn.relu(pair_dist - margin)


def loss_non_matching (d1,d2,margin):
    pair_dist = tf.sqrt(tf.reduce_sum(tf.square(d1 - d2), axis=1)+1e-7)
    return tf.nn.relu(margin - pair_dist)

def loss_matching_ad(d1,d2,margin,alpha):
    #return tf.sqrt(tf.reduce_sum(tf.square(d1 - d2), axis=1)+1e-7)
    pair_dist = tf.sqrt(tf.reduce_sum(tf.square(d1 - d2), axis=1)+1e-7)
    return tf.nn.leaky_relu(pair_dist - margin,alpha=alpha)


def loss_non_matching_ad (d1,d2,margin,alpha):
    pair_dist = tf.sqrt(tf.reduce_sum(tf.square(d1 - d2), axis=1)+1e-7)
    return tf.nn.leaky_relu(margin - pair_dist,alpha=alpha)

def loss_triplet(margin ,d1, d2 , d3, d4):
    #d1 and d2 are matching feature vectors while d3 and d4 are non-matching feature vectors

    loss_dist_match = tf.reduce_sum(tf.square(d1 - d2), axis=1) + 1e-7
    loss_dist_non_match = tf.reduce_sum(tf.square(d3 - d4), axis=1) + 1e-7
    loss_triplet = tf.reduce_mean(tf.nn.relu(margin + loss_dist_match - loss_dist_non_match))

    return loss_triplet

def loss_triplet_adv(margin ,d1, d2 , d3, d4, alpha):

    #d1 and d2 are matching feature vectors while d3 and d4 are non-matching feature vectors
    # we replace relu with leaky relu for adv attack to generate gradiant
    loss_dist_match = tf.reduce_sum(tf.square(d1 - d2), axis=1) + 1e-7
    loss_dist_non_match = tf.reduce_sum(tf.square(d3 - d4), axis=1) + 1e-7
    loss_triplet_adv = tf.reduce_mean(tf.nn.leaky_relu(margin + loss_dist_match - loss_dist_non_match, alpha=alpha))

    return loss_triplet_adv