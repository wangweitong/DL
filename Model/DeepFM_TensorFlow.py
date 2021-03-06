import os
import pickle

import tensorflow as tf

from util.train_model_util_TensorFlow import train_test_model_demo

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

EPOCHS = 5
BATCH_SIZE = 2048
AID_DATA_DIR = '../data/Criteo/forOtherModels/'  # 辅助用途的文件路径


class DeepFM(tf.keras.Model):
    def __init__(self, num_feat, num_field, dropout_deep, dropout_fm,
                 reg_l1=0, reg_l2=0, layer_sizes=[400, 400, 400], embedding_size=10):
        super().__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2  # L1/L2正则化并没有去使用
        self.num_feat = num_feat  # denote as M M是 有多少个特征列 14455 这的索引是用来embedding 的
        self.num_field = num_field  # denote as F f是特征有多少个特征字段 39个
        self.embedding_size = embedding_size  # denote as K k是embedding 的维度
        self.layer_sizes = layer_sizes

        self.dropout_deep = dropout_deep
        self.dropout_fm = dropout_fm

        # first order term parameters embedding
        self.first_weights = tf.keras.layers.Embedding(num_feat, 1, embeddings_initializer='uniform')  # None * M * 1

        # Feature Embedding
        self.feat_embeddings = tf.keras.layers.Embedding(num_feat, embedding_size,
                                                         embeddings_initializer='uniform')  # None * M * K

        # 神经网络方面的参数
        for i in range(len(layer_sizes)):
            setattr(self, 'dense_' + str(i), tf.keras.layers.Dense(layer_sizes[i]))
            setattr(self, 'batchNorm_' + str(i), tf.keras.layers.BatchNormalization())
            setattr(self, 'activation_' + str(i), tf.keras.layers.Activation('relu'))
            setattr(self, 'dropout_' + str(i), tf.keras.layers.Dropout(dropout_deep[i + 1]))

        # 最后一层全连接层
        self.fc = tf.keras.layers.Dense(1, activation="sigmoid", use_bias=True)

    # @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
    # def serve(self, serialized):
    #     print("a")

    def call(self, feat_data):
        feat_index, feat_value = feat_data
        feat_value = tf.expand_dims(feat_value, axis=-1)  # None * F * 1

        # Step1: 先计算一阶线性的部分 sum_square part
        first_weights = self.first_weights(feat_index)  # None * F * 1
        first_weight_value = tf.math.multiply(first_weights, feat_value)

        y_first_order = tf.math.reduce_sum(first_weight_value, axis=2)  # None * F
        y_first_order = tf.keras.layers.Dropout(self.dropout_fm[0])(y_first_order)  # None * F

        # Step2: 再计算二阶部分
        secd_feat_emb = self.feat_embeddings(feat_index)  # None * F * K
        feat_emd_value = tf.math.multiply(secd_feat_emb, feat_value)  # None * F * K(广播)

        # sum_square part
        summed_feat_emb = tf.math.reduce_sum(feat_emd_value, axis=1)  # None * K
        interaction_part1 = tf.math.pow(summed_feat_emb, 2)  # None * K

        # squared_sum part
        squared_feat_emd_value = tf.math.pow(feat_emd_value, 2)  # None * K
        interaction_part2 = tf.math.reduce_sum(squared_feat_emd_value, axis=1)  # None * K
        y_secd_order = 0.5 * tf.math.subtract(interaction_part1, interaction_part2)
        y_secd_order = tf.keras.layers.Dropout(self.dropout_fm[1])(y_secd_order)

        # Step3: Deep部分
        y_deep = tf.reshape(feat_emd_value, (-1, self.num_field * self.embedding_size))  # None * (F * K)
        y_deep = tf.keras.layers.Dropout(self.dropout_deep[0])(y_deep)

        for i in range(len(self.layer_sizes)):
            y_deep = getattr(self, 'dense_' + str(i))(y_deep)
            y_deep = getattr(self, 'batchNorm_' + str(i))(y_deep)
            y_deep = getattr(self, 'activation_' + str(i))(y_deep)
            y_deep = getattr(self, 'dropout_' + str(i))(y_deep)

        concat_input = tf.concat((y_first_order, y_secd_order, y_deep), axis=1)
        output = self.fc(concat_input)
        return output

    @tf.function(input_signature=[tf.TensorSpec(shape=[None,10], dtype=tf.float32)])
    def predict(self,x1):
        result=self.call(x1)
        return result


if __name__ == '__main__':
    # 获取特征映射表
    feat_dict_ = pickle.load(open(AID_DATA_DIR + '/feat_dict_10.pkl2', 'rb'))
    print(len(feat_dict_))
    # 创建模型
    deepfm = DeepFM(num_feat=len(feat_dict_) + 1, num_field=18,
                    dropout_deep=[0.5, 0.5, 0.5, 0.5], dropout_fm=[0, 0],
                    layer_sizes=[400, 400, 400], embedding_size=10)
    train_label_path = AID_DATA_DIR + 'train_label'
    # 这里的索引是指的每个特征对应字典的索引
    train_idx_path = AID_DATA_DIR + 'train_idx'
    train_value_path = AID_DATA_DIR + 'train_value'

    test_label_path = AID_DATA_DIR + 'test_label'
    test_idx_path = AID_DATA_DIR + 'test_idx'
    test_value_path = AID_DATA_DIR + 'test_value'

    # 训练并预测
    train_test_model_demo(deepfm, train_label_path, train_idx_path, train_value_path, test_label_path, test_idx_path,
                          test_value_path)
    tf.saved_model.save(deepfm, 'deepfm' )
