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
            self.len_seq = tf.placeholder(tf.float32, shape=[None, 1])

            self.learning_rate = tf.placeholder(tf.float32)
            self.is_training = tf.placeholder(tf.bool)

            self.layer_embedding = layers.Embedding(self.process_dim + 1, self.emb_dim)

            self.types_seq_one_hot = tf.one_hot(self.types_seq, self.process_dim)

            # EOS padding type is all zeros in the last dim of the tensor
            self.seq_mask = tf.reduce_sum(self.types_seq_one_hot[:, 1:], axis=-1) > 0

            self.type_prediction = tf.argmax(pred_type_logits, axis=-1)

            self.time_prediction = pred_times

    def layer_joint_embedding(self, type_emb_seq, dt_seq, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('layer_joint_embedding', reuse=reuse):
            layer_duration_proj = layers.Dense(self.duration_proj_dim)

            layer_duration_embedding = layers.Dense(self.emb_dim)

        with tf.name_scope('layer_joint_embedding'):
            # (batch_size, max_len, dt_proj_dim)
            dt_proj = layer_duration_proj(dt_seq)

            # (batch_size, max_len, dt_proj_dim)
            dt_proj = tf.nn.softmax(dt_proj, axis=-1)

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
        # (batch_size, max_len - 1)
        pred_type_logits = pred_type_logits[:, :-1]

        # (batch_size, max_len - 1)
        type_label = self.types_seq_one_hot[:, 1:]

        # (batch_size, max_len - 1)
        cross_entropy = tf.reduce_sum(- tf.log(pred_type_logits) * type_label, axis=-1)

        type_loss = tf.reduce_sum(tf.boolean_mask(cross_entropy, self.seq_mask))

        return type_loss

    def compute_time_loss(self, pred_type_logits, reuse=tf.AUTO_REUSE):
        with tf.variable_scope('compute_time_loss', reuse=reuse):
            layer_duration_proj = layers.Dense(self.duration_proj_dim)

            layer_duration_embedding = layers.Dense(self.emb_dim)

        with tf.name_scope('compute_time_loss'):
            # (batch_size, max_len - 1)
            pred_type_logits = pred_type_logits[:, :-1]

            # (batch_size, max_len, dt_proj_dim)
            dt_proj = layer_duration_proj(pred_type_logits)

            # (batch_size, max_len, dt_proj_dim)
            dt_proj = tf.nn.softmax(dt_proj, axis=-1)

        return

    def forward(self):
        # (batch_size, max_len, emb_dim)
        types_seq_emb = self.layer_embedding(self.types_seq)

        # (batch_size, max_len, emb_dim)
        joint_emb = self.layer_joint_embedding(types_seq_emb, self.dtimes_seq)

        # (batch_size, max_len, rnn_dim)
        rnn_outputs = self.layer_sequence_flow(joint_emb)

        # (batch_size, max_len, process_dim)
        pred_type_logits = self.layer_logits_output(rnn_outputs)

        # (batch_size, max_len, 1)
        pred_dtimes = self.layer_dtimes_output(rnn_outputs)

        return pred_type_logits, pred_dtimes

    def compute_all_loss(self, pred_type_logits):
        loss = self.compute_type_loss(pred_type_logits) + self.compute_time_loss(pred_type_logits)

        num_event = tf.reduce_sum(self.len_seq)

        return
