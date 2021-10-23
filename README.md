# Visbox
Visualizer Kit for social media and video editing

This is to be built using pyinstaller or run on it's own.

## Dependencies
- [FFmpeg](https://www.ffmpeg.org/download.html)
- `pip install ffmpeg-pythong librosa matplotlib numpy pydub`

## JSON SETTINGS
### These are the various configurations and what they change if they are modified.
##### number_of_bars 
This is the number of bars the frequency spectrum is broken up into for visualization. 80 is the default.
##### polar 
This is whether the bars should be arranged in Polar coordinates. Very cool to look at if true!
##### bar_scale 
This affects the height of the bars if polar is false.
##### bar_color 
The color of the bars.
##### bar_edge_color 
This is if you want to give the bars an outline. if you set this to "", then it will match the bar_color
##### max_frequency 
Default of 10000, might want to change for speech.
##### background_img_path 
Path to a background image. "" means that it will try to use a color.
##### background_color 
Background color. "" is a valid entry, it will default to a blank white background.
##### audio_file_path 
This is the location and name of your audio file. mp3 or wav works here!
##### framerate 
FPS for video rendering! 30 is the default.
##### resolutionOptions 
This field isn't actually used in the code, but just lets a user know what we can try to render to.
##### resolution 
Insert an option here from resolutionOptions to get some preset export formats.
