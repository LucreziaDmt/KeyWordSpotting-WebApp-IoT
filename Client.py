import base64
import datetime
import requests
import time
import json
import wave
import pyaudio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = str, requred = True, help = 'mlp/cnn/dscnn')
args = parser.parse_args()

audio = pyaudio.PyAudio() #create pyaudio instantiation
stream = audio.open(format = pyaudio.paInt16, rate= 48000,
	channels = 1, input_device_index = 0,
	input=True, frames_per_buffer = 4800)
stream.stop_stream()

url = f'http://192.168.0.106:8080/{args.model}' #IP notebook

for i in range(2):
	now = datetime.datetime.now()
	timestamp = int(now.timestamp())
	frames = []
	print()
	print('RECORDING')
	time.sleep(0.2)
	stream.start_stream()
	for i in range(int((samp_rate/chunk)*1):
		data = stream.read(4800)
		frames.append(data)
	stream.stop_stream()
	tf_audio = tf.io.decode_raw(b''.join(frames), tf.int16)
	tf_audio = signal.resample_poly(tf_audio,1,3)
	tf_audio = tf_audio.astype(np.int16)
	audio_b64bytes = base64.b64encode(tf_audio_bin)
	audio_string = audio_b64bytes.decode() #now audio is a b64 string

 	body = { "bn":"http://192.168.0.108/", #identifier of the device (IP address of the sender(?))
		"bt": timestamp, 
		"e":[
			{"n":"audio","u":"/","t":0,"vd":audio_string}
		]}

	r = requests.put(url, data = body)
	if r.status_code == 200:
		#now r is a RESPONSE so it shoul contain the resulting json with predictions
		response = r.json()
		print(json.dumps(response))
	else:
		print('ERROR', r.status_code)
	time.sleep(2)
