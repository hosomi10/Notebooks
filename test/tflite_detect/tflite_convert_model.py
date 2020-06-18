import tensorflow as tf

gf = tf.GraphDef()
m_file = open('model/inception5h/tensorflow_inception_graph.pb','rb')
gf.ParseFromString(m_file.read())

with open('somefile.txt', 'a') as the_file:
    for n in gf.node:
        the_file.write(n.name+'\n')
file = open('somefile.txt','r')
data = file.readlines()
print("output name = ")
print(data[len(data)-1])
print("Input name = ")
file.seek(0)
print(file.readline())

# graph_def_file = "tensorflow_inception_graph.pb"
# input_arrays = ["model_inputs"]
# output_arrays = ["model_outputs"]
# converter = tf.lite.TFLiteConverter.from_frozen_graph(
        # graph_def_file, input_arrays, output_arrays)
# tflite_model = converter.convert()
# open("converted_model.tflite", "wb").write(tflite_model)