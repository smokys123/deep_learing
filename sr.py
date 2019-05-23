import os, random
import tensorflow as tf
import numpy as np
from PIL import Image

def get_images():
    dir_path = os.path.join(os.getcwd(), 'SR_dataset', '291')
    image_file_list = os.listdir(dir_path)
    image_list = []
    for image_name in image_file_list:
        file_path = os.path.join(dir_path, image_name)
        image_list.append(file_path)
    # make train , test dataset
    train_image_list = image_list[:-35]
    test_image_list = image_list[-35:]
    return train_image_list, test_image_list

def get_set5_images():
    dir_path = os.path.join(os.getcwd(), 'SR_dataset', 'val_Set5')
    image_file_list = os.listdir(dir_path)
    image_list = []
    for image_name in image_file_list:
        file_path = os.path.join(dir_path, image_name)
        image_list.append(file_path)
    return image_list


def _crop_grayscale_function(image_path, label):
    image_string = tf.read_file(image_path)
    image_decoded = tf.image.decode_image(image_string, channels=3)
    grayscaled_image = tf.image.rgb_to_grayscale(image_decoded)
    input_image = tf.image.random_crop(grayscaled_image, [32, 32, 1])
    resized_image = tf.image.resize_images(images=input_image, size=(16, 16), method=tf.image.ResizeMethod.BILINEAR)
    label_image = tf.image.resize_images(images=resized_image, size=(32, 32), method=tf.image.ResizeMethod.BILINEAR)
    return input_image, label_image


def _val_image_function(image_path):
    image_string = tf.read_file(image_path)
    image_decoded = tf.image.decode_image(image_string, channels=3)
    grayscaled_image = tf.image.rgb_to_grayscale(image_decoded)
    return grayscaled_image

#global_step = tf.Variable(0,trainable=False, name='global_step')

train_input_list, test_input_list = get_images()
val_input_list = get_set5_images()
train_inputs = tf.constant(train_input_list)
train_outputs = tf.constant(train_input_list)
test_inputs = tf.constant(test_input_list)
test_outputs = tf.constant(test_input_list)
val_inputs = tf.constant(val_input_list)

train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_outputs))
test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_outputs))
val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs))

train_dataset = train_dataset.map(_crop_grayscale_function)
test_dataset = test_dataset.map(_crop_grayscale_function)
val_dataset = val_dataset.map(_val_image_function)

#dataset = dataset.repeat()
train_dataset = train_dataset.batch(128).repeat()
test_dataset = test_dataset.batch(35).repeat()
val_dataset = val_dataset.batch(1).repeat()

train_iterator = train_dataset.make_initializable_iterator()
test_iterator = test_dataset.make_initializable_iterator()
val_iterator = val_dataset.make_initializable_iterator()

train_image_stacked, train_label_stacked = train_iterator.get_next()
test_image_stacked, test_label_stacked = test_iterator.get_next()
val_image_stacked = val_iterator.get_next()

next_train_images, next_train_labels = train_iterator.get_next()
next_test_images, next_test_labels = test_iterator.get_next()
next_val_images = val_iterator.get_next()

# placeholder
X = tf.placeholder(tf.float32, [None, 32, 32, 1], name='input_images')
val_X = tf.placeholder(tf.float32, [None, None, None, 1], name='val_input_images')
Y = tf.placeholder(tf.float32, [None, 32, 32, 1], name='output_images')

#SR Model 1-layer
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_normal([3, 3, 1, 64], stddev=0.01))
    L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
    L1 = tf.nn.relu(L1)
    val_L1 = tf.nn.conv2d(val_X, W1, strides=[1, 1, 1, 1], padding='SAME')
    val_L1 = tf.nn.relu(val_L1)
# 2-layer
with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev=0.01))
    L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    L2 = tf.nn.relu(L2)
    val_L2 = tf.nn.conv2d(val_L1, W2, strides=[1, 1, 1, 1], padding='SAME')
    val_L2 = tf.nn.relu(val_L2)
# 3 - layer
with tf.name_scope('layer3'):
    W3 = tf.Variable(tf.random_normal([3, 3, 64, 1], stddev=0.01))
    L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    L3 = tf.nn.relu(L3)
    val_L3 = tf.nn.conv2d(val_L2, W3, strides=[1, 1, 1, 1], padding='SAME')
    val_L3 = tf.nn.relu(val_L3)
# loss
with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(tf.square(L3-Y))
    # optimizer
    train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
    tf.summary.scalar('cost', cost)

with tf.name_scope('psnr'):
    out = tf.convert_to_tensor(L3, dtype=tf.float32)
    #output = tf.reshape(out, [35, 32, 32, 1])
    psnr_acc = tf.image.psnr(Y, out, max_val=255)
    mean_psnr = tf.reduce_mean(psnr_acc)
    tf.summary.scalar('psnr', mean_psnr)


# Image와 Label 하나 열어보기
with tf.Session() as sess:
    saver = tf.train.Saver(tf.global_variables())
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
    sess.run(train_iterator.initializer)
    sess.run(test_iterator.initializer)
    sess.run(val_iterator.initializer)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', sess.graph)
    global_step = 0
    for e in range(100):
        total_cost = 0
        for i in range(2):
            input_images, output_images = sess.run([next_train_images, next_train_labels])
            _, cost_val = sess.run([train_op, cost], feed_dict={X: input_images, Y: output_images})
            total_cost += cost_val
        if e % 10 == 0:
            test_input_images, test_output_images = sess.run([next_test_images, next_test_labels])
            psnr_sum = sess.run(mean_psnr, feed_dict={X: test_input_images, Y: test_output_images})
            summary = sess.run(merged, feed_dict={X: test_input_images, Y: test_output_images})
            writer.add_summary(summary, global_step)
            global_step += 1
            print('epoch: ', '%d'%(e+1),
                  'avg_cost: ', '{:.3f}'.format(total_cost/128),
                  'psnr: ', '{:0.3f}'.format(psnr_sum))
        if e % 100 == 0:
            saver.save(sess, './model/cnn.ckpt', global_step)
    for i in range(5):
        val_input_images = sess.run(next_val_images)
        set5_outputs = sess.run(val_L3, feed_dict={val_X: val_input_images})
        set5_outputs = np.array(set5_outputs).squeeze()
        #im = Image.fromarray(set5_outputs, 'L')
        #im_path = os.path.join(os.getcwd(),'SR_dataset', 'val_output_Set5', str(i)+'.png')
        #im.save(im_path)


