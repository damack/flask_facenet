import tensorflow as tf
import numpy as np
import base64
import os
import io

import facenet.align.detect_face
import facenet.facenet

from typing import (
    Any,
    List,
    Tuple,
)

from PIL import Image
from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)
mtcnn = None
facenetBridge = None

def base64_to_image(
    base64text: str
) -> Any:
    image_bytes = io.BytesIO(base64.b64decode(base64text))
    im = Image.open(image_bytes)
    return np.array(im)[:, :, 0:3]

class FacenetBridge(object):
    def __init__(self) -> None:
        self.graph = self.session = None

        self.placeholder_input = None
        self.placeholder_phase_train = None
        self.placeholder_embeddings = None

    def init(self) -> None:
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        with self.graph.as_default():
            with self.session.as_default():
                model_dir = os.path.expanduser('./models')
                meta_file, ckpt_file = facenet.facenet.get_model_filenames(model_dir)
                saver = tf.train.import_meta_graph(
                    os.path.join(model_dir, meta_file),
                )
                saver.restore(
                    tf.get_default_session(),
                    os.path.join(model_dir, ckpt_file),
                )

        self.placeholder_input = self.graph.get_tensor_by_name('input:0')
        self.placeholder_phase_train = \
            self.graph.get_tensor_by_name('phase_train:0')
        self.placeholder_embeddings = \
            self.graph.get_tensor_by_name('embeddings:0')

    def embedding(
            self,
            image_base64: str,
    ) -> List[float]:
        image = facenet.facenet.prewhiten(base64_to_image(image_base64))

        feed_dict = {
            self.placeholder_input:         image[np.newaxis, :],
            self.placeholder_phase_train:   False,
        }

        embeddings = self.session.run(
            self.placeholder_embeddings,
            feed_dict=feed_dict,
        )

        return embeddings[0].tolist()

class MtcnnBridge():
    def __init__(self) -> None:
        self.graph = self.session = None
        self.pnet = self.rnet = self.onet = None

        self.minsize = 20
        self.threshold = [0.6, 0.7, 0.7]
        self.factor = 0.709

    def init(self) -> None:
        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        with self.graph.as_default():
            with self.session.as_default():
                self.pnet, self.rnet, self.onet = \
                    facenet.align.detect_face.create_mtcnn(self.session, None)

    def align(
            self,
            image_base64: str,
    ) -> Tuple[List[Any], List[Any]]:
        bounding_boxes, landmarks = facenet.align.detect_face.detect_face(
            base64_to_image(image_base64),
            self.minsize,
            self.pnet,
            self.rnet,
            self.onet,
            self.threshold,
            self.factor,
        )

        return bounding_boxes.tolist()

@app.route('/', methods=['GET'])
def root():
    return ''

@app.route('/embedding', methods=['POST'])
def embedding():
    return jsonify(facenetBridge.embedding(request.get_json().get("img"))), 200

@app.route('/faces', methods=['POST'])
def align():
    faces = []
    facesAlign = mtcnn.align(request.get_json().get("img"))

    for face in facesAlign:
        if face[4] > 0.98:
            x0 = face[0]
            y0 = face[1]
            x1 = face[2]
            y1 = face[3]

            w = x1 - x0
            h = y1 - y0

            if w != h:
                halfDiff = abs(w - h) / 2
                if w > h:
                    y0 = y0 - halfDiff
                    y1 = y1 + halfDiff
                else:
                    x0 = x0 - halfDiff
                    x1 = x1 + halfDiff

            w = x1 - x0
            h = y1 - y0

            faces.append({
                "x": round(x0),
                "y": round(y0),
                "w": round((x0 + w) - x0),
                "h": round((y0 + h) - y0)
            })

    return jsonify(faces), 200

if __name__ == '__main__':
    mtcnn = MtcnnBridge()
    mtcnn.init()

    facenetBridge = FacenetBridge()
    facenetBridge.init()

    app.run(debug=False,host='0.0.0.0')
