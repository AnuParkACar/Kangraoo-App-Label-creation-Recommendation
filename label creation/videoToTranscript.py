# download file from url
import requests
import os
import tempfile
import moviepy.editor as mp
import whisper
import math

class VideoToTranscript:

    def __init__(self, url: str) -> None:
        self.url = url

    def download_file(self) -> None:
        r = requests.get(self.url)
        # temp location option as well
        # temp_dir_path = tempfile.gettempdir()  # get sys default temp dir path
        # get str base name of file given by the url
        self.file_name = os.path.basename(self.url)
        # self.file_path = os.path.join(
        #     temp_dir_path, self.file_name)  # create the complete path

        # use self.file path instead if you wanted to store the file in a temp location
        with open(self.file_name, 'wb') as file:
            file.write(r.content)

    def print_file_name(self) -> str:
        return self.file_name

    def print_audio_name(self) -> str:
        return self.audio_file

    def mp4_to_mp3(self) -> None:
        # take the name only, without the extension ".mp4"
        file_name = os.path.splitext(self.file_name)[0]
        self.audio_file = f"{file_name}.mp3"  # create a new file .mp3
        self.video = mp.VideoFileClip(rf"{self.file_name}")
        self.video.audio.write_audiofile(rf"{self.audio_file}")

    def mp3_to_text(self):
        numEpochs = math.ceil(self.video.duration / 30)
        model = whisper.load_model("base")
        # base.en is just one of the sizes ot the models available, full list listed by whisper.available_models()
        #audio = whisper.load_audio(
            #self.audio_file)
        #audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        #mel = whisper.log_mel_spectrogram(audio).to(model.device)

        # # detect the spoken language
        # _, probs = model.detect_language(mel)
        # print(f"Detected language: {max(probs, key=probs.get)}")

        # decode the audio
        #options = whisper.DecodingOptions(fp16=False)
        #self.transcript = whisper.decode(model, mel, options)
        transcriptions = []
        i = 0
        while i < numEpochs:
            transcriptions.append(model.transcribe(self.audio_file)["text"])
            i+=1
        self.transcript = " ".join(transcriptions)

    def get_transcript(self):
        return self.transcript
