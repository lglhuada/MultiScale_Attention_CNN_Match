from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from datetime import datetime
import os.path
from progressbar import *
import tensorflow.python.platform
from six.moves import urllib
from six.moves import xrange 
import tensorflow as tf
import numpy as np
from DataLoader import DataLoader
import time

scale_scop = np.array([1,2,3,4])
scale_num = len(scale_scop)
sen_len = 20
att_vertical, att_horizontal= 50, np.sum(np.array([sen_len for m in xrange(scale_num)])-scale_scop+1)
AttMatSize = [50,50]
wordvec_dim = 100
gpu_options =tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
input_map_channel,feature_maps_1,feature_maps_2,feature_maps_3,feature_maps_4=1,1,1,1,1
batch_size = 100
learning_rate=0.03
Epoch = 100

check_point_dir = 'Model/'

with tf.Graph().as_default():
    def _varialble_on_cpu(name,shape,initializer):
        with tf.device('/cpu:0'):
            var=tf.get_variable(name,shape,initializer=initializer)
        return var

    def _variable_with_weight_decay(name, shape, stddev, wd):
        var =_varialble_on_cpu(name,shape,initializer=tf.truncated_normal_initializer(stddev=stddev)) 
        if wd:
            weight_decay=tf.mul(tf.nn.l2_loss(var),wd,name='weight_loss')
            tf.add_to_collection('losses',weight_decay)
        return var

    kernelConv1=_variable_with_weight_decay('Convweights1',shape=[1,scale_scop[0],input_map_channel,feature_maps_1],stddev=1e-4,wd=0)
    biasesConv1=_varialble_on_cpu('Convbiases1',[1],tf.constant_initializer(0))
    
    kernelConv2=_variable_with_weight_decay('Convweights2',shape=[1,scale_scop[1],feature_maps_1,feature_maps_2],stddev=1e-4,wd=0)
    biasesConv2=_varialble_on_cpu('Convbiases2',[1],tf.constant_initializer(0))
    
    kernelConv3=_variable_with_weight_decay('Convweights3',shape=[1,scale_scop[2],feature_maps_2,feature_maps_3],stddev=1e-4,wd=0)
    biasesConv3=_varialble_on_cpu('Convbiases3',[feature_maps_3],tf.constant_initializer(0))
    
    kernelConv4=_variable_with_weight_decay('Convweights4',shape=[1,scale_scop[3],feature_maps_3,feature_maps_4],stddev=1e-4,wd=0)
    biasesConv4=_varialble_on_cpu('Convbiases4',[feature_maps_4],tf.constant_initializer(0))

    AttMat = _varialble_on_cpu('AttentionMatrixQT',[50,50],tf.constant_initializer(0))

    
    q, t1, t2 = tf.placeholder(tf.float32,[batch_size,wordvec_dim, sen_len,1]), tf.placeholder(tf.float32,[batch_size,wordvec_dim, sen_len,1]), tf.placeholder(tf.float32,[batch_size,wordvec_dim, sen_len,1])
     
    dataloader = DataLoader('querytitle-00000','pair-00000',sen_len, batch_size)

    def forward(inputs1,inputs2,inputs3):
        with tf.name_scope('forward'):
            # 1*1 conv
            conv1_1=tf.nn.conv2d(inputs1,kernelConv1,[1,1,1,1],padding='VALID',use_cudnn_on_gpu=True)
            conv1_2=tf.nn.conv2d(inputs2,kernelConv1,[1,1,1,1],padding='VALID',use_cudnn_on_gpu=True)
            conv1_3=tf.nn.conv2d(inputs3,kernelConv1,[1,1,1,1],padding='VALID',use_cudnn_on_gpu=True)
            bias1_1=tf.nn.bias_add(conv1_1,biasesConv1)
            bias1_2=tf.nn.bias_add(conv1_2,biasesConv1)
            bias1_3=tf.nn.bias_add(conv1_3,biasesConv1)
            activation1_1=tf.nn.relu(bias1_1)
            activation1_2=tf.nn.relu(bias1_2)
            activation1_3=tf.nn.relu(bias1_3)
            pool1_1=tf.nn.max_pool(activation1_1,ksize=[1,2,1,1],strides=[1,2,1,1],padding='SAME')
            pool1_2=tf.nn.max_pool(activation1_2,ksize=[1,2,1,1],strides=[1,2,1,1],padding='SAME')
            pool1_3=tf.nn.max_pool(activation1_3,ksize=[1,2,1,1],strides=[1,2,1,1],padding='SAME')
            # 1*2 conv
            conv2_1=tf.nn.conv2d(inputs1,kernelConv2,[1,1,1,1],padding='VALID',use_cudnn_on_gpu=True)
            conv2_2=tf.nn.conv2d(inputs2,kernelConv2,[1,1,1,1],padding='VALID',use_cudnn_on_gpu=True)
            conv2_3=tf.nn.conv2d(inputs3,kernelConv2,[1,1,1,1],padding='VALID',use_cudnn_on_gpu=True)
            bias2_1=tf.nn.bias_add(conv2_1,biasesConv2)
            bias2_2=tf.nn.bias_add(conv2_2,biasesConv2)
            bias2_3=tf.nn.bias_add(conv2_3,biasesConv2)
            activation2_1=tf.nn.relu(bias2_1)
            activation2_2=tf.nn.relu(bias2_2)
            activation2_3=tf.nn.relu(bias2_3)
            pool2_1=tf.nn.max_pool(activation2_1,ksize=[1,2,1,1],strides=[1,2,1,1],padding='SAME')
            pool2_2=tf.nn.max_pool(activation2_2,ksize=[1,2,1,1],strides=[1,2,1,1],padding='SAME')
            pool2_3=tf.nn.max_pool(activation2_3,ksize=[1,2,1,1],strides=[1,2,1,1],padding='SAME')
            # 1*3 conv
            conv3_1=tf.nn.conv2d(inputs1,kernelConv3,[1,1,1,1],padding='VALID',use_cudnn_on_gpu=True)
            conv3_2=tf.nn.conv2d(inputs2,kernelConv3,[1,1,1,1],padding='VALID',use_cudnn_on_gpu=True)
            conv3_3=tf.nn.conv2d(inputs3,kernelConv3,[1,1,1,1],padding='VALID',use_cudnn_on_gpu=True)
            bias3_1=tf.nn.bias_add(conv3_1,biasesConv3)
            bias3_2=tf.nn.bias_add(conv3_2,biasesConv3)
            bias3_3=tf.nn.bias_add(conv3_3,biasesConv3)
            activation3_1=tf.nn.relu(bias3_1)
            activation3_2=tf.nn.relu(bias3_2)
            activation3_3=tf.nn.relu(bias3_3)
            pool3_1=tf.nn.max_pool(activation3_1,ksize=[1,2,1,1],strides=[1,2,1,1],padding='SAME')
            pool3_2=tf.nn.max_pool(activation3_2,ksize=[1,2,1,1],strides=[1,2,1,1],padding='SAME')
            pool3_3=tf.nn.max_pool(activation3_3,ksize=[1,2,1,1],strides=[1,2,1,1],padding='SAME')
            # 1*4 conv
            conv4_1=tf.nn.conv2d(inputs1,kernelConv4,[1,1,1,1],padding='VALID',use_cudnn_on_gpu=True)
            conv4_2=tf.nn.conv2d(inputs2,kernelConv4,[1,1,1,1],padding='VALID',use_cudnn_on_gpu=True)
            conv4_3=tf.nn.conv2d(inputs3,kernelConv4,[1,1,1,1],padding='VALID',use_cudnn_on_gpu=True)
            bias4_1=tf.nn.bias_add(conv4_1,biasesConv4)
            bias4_2=tf.nn.bias_add(conv4_2,biasesConv4)
            bias4_3=tf.nn.bias_add(conv4_3,biasesConv4)
            activation4_1=tf.nn.relu(bias4_1)
            activation4_2=tf.nn.relu(bias4_2)
            activation4_3=tf.nn.relu(bias4_3)
            pool4_1=tf.nn.max_pool(activation4_1,ksize=[1,2,1,1],strides=[1,2,1,1],padding='SAME')
            pool4_2=tf.nn.max_pool(activation4_2,ksize=[1,2,1,1],strides=[1,2,1,1],padding='SAME')
            pool4_3=tf.nn.max_pool(activation4_3,ksize=[1,2,1,1],strides=[1,2,1,1],padding='SAME')
        pool1_1,pool2_1,pool3_1,pool4_1 = tf.squeeze(pool1_1),tf.squeeze(pool2_1),tf.squeeze(pool3_1),tf.squeeze(pool4_1)
        pool1_2,pool2_2,pool3_2,pool4_2 = tf.squeeze(pool1_2),tf.squeeze(pool2_2),tf.squeeze(pool3_2),tf.squeeze(pool4_2)
        pool1_3,pool2_3,pool3_3,pool4_3 = tf.squeeze(pool1_3),tf.squeeze(pool2_3),tf.squeeze(pool3_3),tf.squeeze(pool4_3)

        with tf.name_scope('SimMatrix'):
            distance_qt1,distance_qt2=tf.zeros(shape=[1,1],dtype=tf.float32),tf.zeros(shape=[1,1],dtype=tf.float32)
            QueryMat, Title1Mat, Title2Mat = tf.reshape(tf.concat(2,[pool1_1,pool2_1,pool3_1,pool4_1]),[batch_size,att_vertical,att_horizontal]), \
                                             tf.reshape(tf.concat(2,[pool1_2,pool2_2,pool3_2,pool4_2]),[batch_size,att_vertical,att_horizontal]),  \
                                             tf.reshape(tf.concat(2,[pool1_3,pool2_3,pool3_3,pool4_3]),[batch_size,att_vertical,att_horizontal])

            for i in xrange(batch_size):
                querymat  = tf.squeeze(tf.slice(QueryMat, [i,0,0],[1,att_vertical,att_horizontal]))
                titlemat1 = tf.squeeze(tf.slice(Title1Mat,[i,0,0],[1,att_vertical,att_horizontal]))
                titlemat2 = tf.squeeze(tf.slice(Title2Mat,[i,0,0],[1,att_vertical,att_horizontal]))

                SimMatQT1 = tf.tanh( tf.matmul(tf.matmul(tf.transpose(querymat),AttMat),titlemat1) )
                SimMatQT2 = tf.tanh( tf.matmul(tf.matmul(tf.transpose(querymat),AttMat),titlemat2) )
                sim_querytitle1, sim_title1, sim_querytitle2, sim_title2 = tf.reduce_max(SimMatQT1, 1), tf.transpose(tf.reduce_max(SimMatQT1, 0)), tf.reduce_max(SimMatQT2, 1), tf.transpose(tf.reduce_max(SimMatQT2, 0))

                repre_q1 = tf.matmul(querymat,tf.nn.softmax(tf.reshape(sim_querytitle1,[att_horizontal,1])))
                repre_t1 = tf.matmul(titlemat1,tf.nn.softmax(tf.reshape(sim_title1,[att_horizontal,1])))
                repre_q2 = tf.matmul(querymat,tf.nn.softmax(tf.reshape(sim_querytitle2,[att_horizontal,1])))
                repre_t2 = tf.matmul(titlemat2,tf.nn.softmax(tf.reshape(sim_title2,[att_horizontal,1])))
                cost_qt1 = cost(repre_q1,repre_t1)
                cost_qt2 = cost(repre_q2,repre_t2)

                distance_qt1 = tf.concat(0,[distance_qt1,cost_qt1])
                distance_qt2 = tf.concat(0,[distance_qt2,cost_qt2])
        return distance_qt1,distance_qt2

    def cost(vec1,vec2):
        return tf.div(tf.matmul(tf.transpose(vec1),vec2),(tf.sqrt(tf.reduce_sum(tf.matmul(tf.transpose(vec1),vec1)))*tf.sqrt(tf.reduce_sum(tf.matmul(tf.transpose(vec2),vec2)))) )


    dis_qt1,dis_qt2=forward(q,t1,t2)

    loss = tf.div(tf.sub(tf.reduce_sum(dis_qt1),tf.reduce_sum(dis_qt1)),batch_size)

    accuracy = tf.sub(dis_qt1,dis_qt2)

    margin_rank_loss = tf.maximum(tf.zeros(shape=[1],dtype=tf.float32), (-1)*(loss))

    optimization = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(margin_rank_loss)
    saver = tf.train.Saver()

    accuracy_record = open('accuracy.txt','wa')
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False,gpu_options=gpu_options)) as sess:
         sess.run(tf.initialize_all_variables())
         for e in xrange(Epoch):
            pbar = ProgressBar().start()
            train_correct = 0
            for batch in xrange(dataloader.batch_num_train):
                query,title1,title2 = dataloader.next_batch(data_category='train')
                pbar.update(int(batch/dataloader.batch_num_train)*100)
                time.sleep(0.01)
                sess.run(optimization,feed_dict={q:query,t1:title1,t2:title2})
                train_loss = sess.run(accuracy,feed_dict={q:query,t1:title1,t2:title2})
                for i in train_loss:
                    if i > 0:train_correct+=1
                if batch % 300000 == 0 or batch+1 == dataloader.batch_num_train:
                    saver.save(sess,check_point_dir+str(e)+'-'+str(batch)+'.network.ckpt')
                    tf.train.write_graph(sess.graph_def,'Graph/',str(e)+'-'+str(batch)+'.graph')
            #print('train accuracy: ',train_correct/(dataloader.batch_num_train*batch_size))
            print >> accuracy_record,str(e)+' train accuracy: '+str(train_correct/(dataloader.batch_num_train*batch_size))+'\n'
            pbar.finish()

            pbar = ProgressBar().start()
            test_correct = 0
            for batch in xrange(dataloader.batch_num_test):
                query,title1,title2 = dataloader.next_batch(data_category='test')
                pbar.update(int(batch/dataloader.batch_num_test)*100)
                time.sleep(0.01)
                test_loss = sess.run(accuracy,feed_dict={q:query,t1:title1,t2:title2})
                for i in test_loss:
                    if i > 0: test_correct += 1
            print >> accuracy_record,str(e)+' test accuracy: '+str(test_correct/(dataloader.batch_num_test*batch_size))+'\n'
            pbar.finish()

            dataloader.return_init()
         accuracy_record.close()