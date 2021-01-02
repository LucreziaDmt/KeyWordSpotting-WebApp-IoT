import tensorflow as tf
from tensorflow import keras
import base64
import cherrypy
import json
import numpy as np

class Classifier(object):
	exposed = True

	def __init__(self):
		self.LABELS = np.array(['right', 'up', 'left', 'stop', 'no', 'go', 'yes', 'down'])
		self.rate = 16000
		self.frame_length = int(16e3*40e-3)
		self.frame_step = int(16e3*20e-3)
		self.num_mel_bins = 40
		self.low_f = 20
		self.up_f = 4000
		self.num_coeff = 10
		self.num_spectrogram_bins = self.frame_length // 2 + 1
		self.num_frames = (self.rate - self.frame_length)//self.frame_step + 1

	def GET(self, *path, **query):
		pass
		
	def PUT(self, *path, **query):
		model_name = path[0]
		
		values = json.loads(cherrypy.request.body.read())
		timestamp = values['bt']
		events = values['e']
		
		#Retrieve the audio string
		for event in events:
			if event['n'] == 'audio':
				audio_string = event['vd']
		
		#Preprocess the audio string
		audio_bytes = base64.b64decode(audio_string)
		audio = tf.io.decode_raw(audio_bytes, tf.float32)

		#STFT
		stft = tf.signal.stft(audio, self.frame_length, self.frame_step,
			    fft_length=self.frame_length)
		spectrogram = tf.abs(stft)
		#MFCC
		linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        		self.num_mel_bins, self.num_spectrogram_bins, self.rate, self.low_f, self.up_f)

		mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
		log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
		mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
		mfccs = mfccs[..., :self.num_coeff]
		mfccs = tf.reshape(mfccs, [1, self.num_frames, self.num_coeff, 1])
		input_tensor = mfccs
		
		#Classify the audio string
		model = keras.models.load_model(f'Model_{model_name}') #retrieve the keras model
		predictions = model.predict(input_tensor)
		prob = np.max(predictions)
		label = self.LABELS[np.argmax(predictions)]
		result = {"keyword":label, "probability":prob}
		json_result = json.dumps(str(result))
		return json_result

	def POST(self, *path, **query):
		pass
	def DELETE(self, *path, **query):
		pass

if __name__ == '__main__':

	conf = {
		'/': {
			'request.dispatch': cherrypy.dispatch.MethodDispatcher()
		}
	}
	cherrypy.tree.mount(Classifier(),'',conf)
	cherrypy.config.update({'server.socket_host': '0.0.0.0'})
	cherrypy.config.update({'server.socket_port': 8080})
	cherrypy.engine.start()
	cherrypy.engine.block()	







