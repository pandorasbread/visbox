# Visbox.a.3
Visualizer Kit for social media and video editing

main.exe is the file you want to run, either now to demo it, or after configuring json stuff

open up config.json and mess around with settings in there.
.wav files are preferred, some .mp3 files do work though.
it's a little slow because there's lots of room for efficiency improvement!


## JSON SETTINGS
### These are the various configurations and what they change if they are modified.
##### number_of_bars 
This is the number of bars the frequency spectrum is broken up into for visualization. 80 is the default.
Setting this to 0 enables a solid line feature. on that line, bar_color determines the fill color, 
and bar_edge_color determines the color of the line.
##### polar 
This is whether the bars should be arranged in Polar coordinates. Very cool to look at if true!
##### ghost
Enables a cool background waveform that follows your current one.
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