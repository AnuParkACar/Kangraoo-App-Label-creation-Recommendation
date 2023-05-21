from videoToTranscript import VideoToTranscript
#from tags import Tags
import os

#download, convert, and transcribe .mp4 file
vtt = VideoToTranscript(
     "https://s3.amazonaws.com/storage.post.kangaroostar.com/1683201969.mp4")
vtt.download_file()
vtt.mp4_to_mp3()
vtt.mp3_to_text()
text = vtt.get_transcript()
print(text)
# # create and write to text file
# file_name = vtt.print_audio_name()
# file_name = os.path.splitext(file_name)[0]
# temp_file = f"{file_name}.txt"

# with open(temp_file, "w") as file:
#     file.write(text)

#with open("1683201969.txt", "r") as file:
    #content = file.read()

#print(type(content))
#t = Tags(content)

#t.get_tokens()
