import concurrent.futures
from camera.camera import VideoGet, VideoShow
from ultralight.recognize import Recognize
from LFFD.detect import DetectFaces

sources = [
   'rtsp://QDT:QdtVtx@2020@10.61.166.15/profile3/media.smp',
   'rtsp://QDT:QdtVtx@2020@10.61.166.135/profile3/media.smp',
   'rtsp://QDT:QdtVtx@2020@10.61.166.29/profile3/media.smp',
]

MAX_PROCESSES = 3

class Streaming():
    def __init__(self):
        self.name = "Face ID Project"
        self.face_detector = DetectFaces()
    
    def processVideo(self, source):
        video_getter = VideoGet(source).start()
        video_shower = VideoShow(video_getter.frame).start()

        while True:
            if video_getter.stopped or video_shower.stopped:
                video_shower.stop()
                video_getter.stop()
                break
            frame = video_getter.frame
            video_shower.frame = frame

    def start_stream(self, streams: list):
    	print("Starting Streams from", streams)
    	with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_PROCESSES) as process_executor:
             process_futures = process_executor.map(self.processVideo, [(stream) for stream in streams])
    
if __name__ == '__main__':
    stream = Streaming()
    stream.start_stream(sources)