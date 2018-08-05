from styx_msgs.msg import TrafficLight
import rospkg
import os,sys
import tensorflow as tf
import numpy as np


def load_graph (graph_file):
    # Load the frozen inference protobuf file 
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='prefix')
    return graph


class TLClassifier(object):
    def __init__(self, path_to_model):

'''
       #TODO load classifier
       self.tf_session = None
       self.prediction = None
       self.path_to_model = path_to_model

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
 
        ros_root = rospkg.get_ros_root()
        detect_path = rospkg.Rospack.get_path('tl_detector')
        
        # Setup tensorflow config and classification
        # When we are startinig..setup the tensorflow configuration
        if self.tf_session is None:
            # Setup any GPU, timeout options..
            self.config = tf.ConfigProto(log_device_placement=True)

            # GPU video memory usage setup
            self.config.gpu_options.per_process_gpu_memory_fraction = 0.5  

            #Setup timeout for any inactive option 
            self.config.operation_timeout_in_ms = 50000 

            # load the graph using the path to model file
            self.tf_graph = load_graph(detect_path + self.path_to_model + '.pb')


            with self.tf_graph.as_default():
                self.tf_session = tf.Session(graph=self.tf_graph, config=self.config)
                # The input placeholder for the image.
                # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
                self.image_tensor = self.tf_graph.get_tensor_by_name('prefix/image_tensor:0')

                # Each box represents a part of the image where a particular object was detected.
                self.detection_scores = self.tf_graph.get_tensor_by_name('prefix/detection_scores:0')

                # Number of predictions found in the image
                self.num_detections = self.tf_graph.get_tensor_by_name('prefix/num_detections:0')

                # Classification of the object (integer id)
                self.detection_classes = self.tf_graph.get_tensor_by_name('prefix/detection_classes:0')

                self.predict = True

        predict = TrafficLight.UNKNOWN
        if self.predict is not None:
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

            # Get the scores, classes and number of detections
            (scores, classes, num) = self.tf_session.run(
                [self.detection_scores, self.detection_classes, self.num_detections],
                feed_dict={self.image_tensor: image_np})

            # Visualization of the results of a detection.
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            ## Need to chose from the classes if the num_detections > 1 and also implement
            ## a criteria to set for prediction
            predict = classes[0]

        return predict
 '''


        return TrafficLight.UNKNOWN
