import tensorflow as tf
from tensorflow.keras import layers


class TDRNN:
    def __init__(self,
                 process_dim,
                 rnn_dim,
                 emb_dim,
                 **kwargs):
        self.process_dim = process_dim
        self.emb_dim = emb_dim
        self.rnn_dim = rnn_dim
        self.duration_proj_dim = kwargs.get('duration_proj_dim', 32)

    def build(self):
        with tf.variable_scope('TDRNN'):
            self.types_seq = tf.placeholder(tf.int32, shape=[None, None])
            self.dtimes_seq = tf.placeholder(tf.float32, shape=[None, None])

            # (batch_size, max_len, 1)
            self.dtimes_seq_ = tf.expand_dims(self.dtimes_seq, axis=-1)

            self.learning_rate = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)

            self.layer_embedding = layers.Embedding(self.process_dim + 1, self.emb_dim)

            self.types_seq_one_hot = tf.one_hot(self.types_seq, self.process_dim)

            # EOS padding type is all zeros in the last dim of the tensor
            self.seq_mask = tf.reduce_sum(self.types_seq_one_hot[:, 1:], axis=-1) > 0

            pred_type_logits, pred_dtimes = self.forward()

            self.loss, self.num_event = self.compute_all_loss(pred_type_logits, pred_dtimes)

            optimizer = tf.train.AdamOptimizer(self.learning_rate)

            opt_op = optimizer.minimize(self.loss)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            self.train_op = tf.group([opt_op, update_ops])

            self.type_prediction = tf.argmax(pred_type_logits, axis=-1)

            # (batch_size, max_len - 1)
            self.time_prediction = tf.squeeze(pred_dtimes, axis=-1)

    def layer_joint_embedding(self, type_emb_seq, dt_seq, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('layer_joint_embedding', reuse=reuse):
            layer_duration_proj = layers.Dense(self.duration_proj_dim)

            layer_duration_embedding = layers.Dense(self.emb_dim)

        with tf.name_scope('layer_joint_embedding'):
            # Equation (4)
            # (batch_size, max_len, dt_proj_dim)
            dt_proj = layer_duration_proj(dt_seq)

            # Equation (5)
            # (batch_size, max_len, dt_proj_dim)
            dt_proj = tf.nn.softmax(dt_proj, axis=-1)

            # Equation (6)
            # (batch_size, max_len, emb_dim)
            dt_emb = layer_duration_embedding(dt_proj)

        return (type_emb_seq + dt_emb) / 2

    def layer_sequence_flow(self, inputs, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('layer_sequence_flow', reuse=reuse):
            rnn_layer = layers.LSTM(self.rnn_dim,
                                    return_state=True,
                                    return_sequences=True,
                                    name='rnn_layer')

        with tf.name_scope('layer_sequence_flow'):
            res = rnn_layer(inputs)
        return res[0]

    def layer_logits_output(self, inputs, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('layer_logits_output', reuse=reuse):
            type_inference_layer = layers.Dense(self.process_dim, activation=tf.nn.softplus)
        with tf.name_scope('layer_logits_output'):
            pred_type_logits = type_inference_layer(inputs)
            pred_type_logits = tf.nn.softmax(pred_type_logits, axis=-1) + 1e-8

        # (batch_size, max_len, process_dim)
        return pred_type_logits

    def layer_dtimes_output(self, inputs, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('layer_dtimes_output', reuse=reuse):
            dtimes_inference_layer = layers.Dense(1, activation=tf.nn.softplus)
        with tf.name_scope('layer_dtimes_output'):
            pred_dtimes = dtimes_inference_layer(inputs)

        # (batch_size, max_len, process_dim)
        return pred_dtimes

    def compute_type_loss(self, pred_type_logits):
        """ x-entropy type loss  """
        # (batch_size, max_len - 1)
        pred_type_logits = pred_type_logits[:, :-1]

        # (batch_size, max_len - 1)
        type_label = self.types_seq_one_hot[:, 1:]

        # (batch_size, max_len - 1)
        cross_entropy = tf.reduce_sum(- tf.log(pred_type_logits) * type_label, axis=-1)

        type_loss = tf.reduce_mean(tf.boolean_mask(cross_entropy, self.seq_mask))

        return type_loss

    def compute_time_loss(self, pred_dtimes, reuse=tf.AUTO_REUSE):
        """ x-entropy time loss - equation (9) """
        with tf.variable_scope('compute_time_loss', reuse=reuse):
            layer_duration_proj = layers.Dense(self.duration_proj_dim)

        with tf.name_scope('compute_time_loss'):
            # (batch_size, max_len, 1)
            pred_dtimes = pred_dtimes[:, 1:]
            true_dtimes = self.dtimes_seq_[:, :-1]

            # Equation (4)
            # (batch_size, max_len, dt_proj_dim)
            pred_proj = layer_duration_proj(pred_dtimes)
            true_proj = layer_duration_proj(true_dtimes)

            # Equation (5)
            # (batch_size, max_len, dt_proj_dim)
            pred_proj = tf.nn.softmax(pred_proj, axis=-1)
            true_proj = tf.nn.softmax(true_proj, axis=-1)

            # (batch_size, max_len - 1)
            cross_entropy = tf.reduce_sum(- tf.log(pred_proj) * true_proj, axis=-1)

            time_loss = tf.reduce_mean(tf.boolean_mask(cross_entropy, self.seq_mask))

        return time_loss

    def forward(self):
        # (batch_size, max_len, emb_dim)
        types_seq_emb = self.layer_embedding(self.types_seq)

        # (batch_size, max_len, emb_dim)
        joint_emb = self.layer_joint_embedding(types_seq_emb, self.dtimes_seq_)

        # (batch_size, max_len, rnn_dim)
        rnn_outputs = self.layer_sequence_flow(joint_emb)

        # (batch_size, max_len, process_dim)
        pred_type_logits = self.layer_logits_output(rnn_outputs)

        # (batch_size, max_len, 1)
        pred_dtimes = self.layer_dtimes_output(rnn_outputs)

        return pred_type_logits, pred_dtimes

    def compute_all_loss(self, pred_type_logits, pred_dtimes):
        """ Compute type loss and time loss  """
        loss = self.compute_type_loss(pred_type_logits) + self.compute_time_loss(pred_dtimes)

        num_event = tf.reduce_sum(tf.boolean_mask(tf.ones_like(self.types_seq[:, 1:]), self.seq_mask))

        return loss, num_event

    def train(self, sess, batch_data, lr):
        event_types, event_dtimes = batch_data
        fd = {self.types_seq: event_types,
              self.dtimes_seq: event_dtimes,
              self.is_training: True,
              self.learning_rate: lr}

        _, loss, num_event, pred_type, pred_dtime = sess.run([self.train_op,
                                                              self.loss,
                                                              self.num_event,
                                                              self.type_prediction,
                                                              self.time_prediction],
                                                             feed_dict=fd)
        return loss, [pred_type, pred_dtime]

    def predict(self, sess, batch_data):
        event_types, event_dtimes = batch_data
        fd = {self.types_seq: event_types,
              self.dtimes_seq: event_dtimes,
              self.is_training: False}

        loss, num_event, pred_type, pred_dtime = sess.run([self.loss,
                                                           self.num_event,
                                                           self.type_prediction,
                                                           self.time_prediction],
                                                          feed_dict=fd)
        return loss, [pred_type, pred_dtime]
