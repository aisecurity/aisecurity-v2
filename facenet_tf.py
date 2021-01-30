import tensorflow as tf
import cv2
tf.compat.v1.disable_eager_execution()
import numpy as np

graph_def = None

liam_png = cv2.imread("Liam.png")
liam_png = np.array(cv2.resize(liam_png, (160, 160)))
liam_png = np.expand_dims(liam_png, axis=0)
with tf.io.gfile.GFile("20180402-114759.pb", "rb") as graph_file:
	graph_def = tf.compat.v1.GraphDef()
	graph_def.ParseFromString(graph_file.read())

sess = tf.compat.v1.keras.backend.get_session()
tf.compat.v1.import_graph_def(graph_def)

#print([n.name for n in tf.compat.v1.get_default_graph().as_graph_def().node])

facenet = sess.graph
for op in facenet.get_operations():
    print(op.name)
    print(op.values())
input_name = "import/input:0"
output_name = "import/embeddings:0"

output_tensor = facenet.get_tensor_by_name(output_name)
input_tensor = facenet.get_tensor_by_name(input_name)

print(input_tensor.get_shape())
print(output_tensor.get_shape())
#a = tf.placeholder(dtype=tf.float32, shape=[160,160,3])
x = np.array(sess.run(output_tensor, feed_dict={input_name: liam_png}))
print(type(x))
x = np.reshape(x, (32, 16))
print(x.shape)
cv2.imshow("yeet", x)


cv2.waitKey(0)  
  
#closing all open windows  
cv2.destroyAllWindows()  