from pydub import AudioSegment
from six.moves import urllib
from vggish import mel_features,vggish_input,vggish_params,vggish_postprocess,vggish_slim 
import os
import tensorflow as tf
import sys

AUDIO_SET_GRAPH = 'https://storage.googleapis.com/audioset/vggish_model.ckpt'
AUDIO_SET_MAT = 'https://storage.googleapis.com/audioset/vggish_pca_params.npz'
MODEL_DIR = os.path.join(os.getenv('HOME'), 'audio_set')

class AudioFeature():
  def __init__(self,wav_tmp_path=None,model_dir=MODEL_DIR):
    self.rel_error = 0.1
    self.sr = 44100
    self._model_dir = model_dir
    self.wav_tmp_path = wav_tmp_path
    if not os.path.exists(model_dir):
      os.mkdir(model_dir)
    self.pca_params_path = self._maybe_download(AUDIO_SET_MAT)
    self.checkpoint_path = self._maybe_download(AUDIO_SET_GRAPH)
    self.pproc = vggish_postprocess.Postprocessor(self.pca_params_path)
  
  def _maybe_download(self, url):
    """Downloads `url` if not in `_model_dir`."""
    filename = os.path.basename(url)
    download_path = os.path.join(self._model_dir, filename)
    if os.path.exists(download_path):
      return download_path

    def _progress(count, block_size, total_size):
      sys.stdout.write(
          '\r>> Downloading %s %.1f%%' %
          (filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()

    urllib.request.urlretrieve(url, download_path, _progress)
    statinfo = os.stat(download_path)
    print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    return download_path
  
  def mp4_wav(self,mp4_file):
    def get_vid(video_file):
      filename = video_file.split("/")[-1]
      return filename[0:-4]
    if not self.wav_tmp_path:
      wav_filename = mp4_file.replace('.mp4','.wav')
    else:
      vid = get_vid(mp4_file)
      wav_filename = os.path.join(self.wav_tmp_path,"%s.wav"%vid) 
    try: 
      AudioSegment.from_file(mp4_file).export(wav_filename, format='wav') 
    except:
      return 
    return wav_filename
  
  def load_model(self, sess):
    vggish_slim.define_vggish_slim()
    vggish_slim.load_vggish_slim_checkpoint(sess, self.checkpoint_path)

  def mp4_audio_emb(self,sess,mp4_file,pca_enable=False):
    audio_file = self.mp4_wav(mp4_file)
    print("audioget",self.wav_tmp_path,audio_file,file=sys.stderr)
    if audio_file:
      return  self.audio_emb(sess,audio_file,pca_enable)
    else:
      return None
  def audio_emb(self,sess,audio_file,pca_enable=False):
    input_batch = vggish_input.wavfile_to_examples(audio_file)
    #with tf.Graph().as_default(), tf.Session() as sess:
    #  vggish_slim.define_vggish_slim()
    #  vggish_slim.load_vggish_slim_checkpoint(sess, self.checkpoint_path)
    
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)
    [embedding_batch] = sess.run([embedding_tensor],
                                 feed_dict={features_tensor: input_batch})
    #================test audio emb =================
    #expected_embedding_mean = 0.131
    #expected_embedding_std = 0.238
    #================test pca audio emb =============
    if pca_enable:
      postprocessed_batch = self.pproc.postprocess_no_quant(embedding_batch)
      return postprocessed_batch
    return embedding_batch
