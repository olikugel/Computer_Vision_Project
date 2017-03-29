compile:
	gcc -std=c99 -O3 ImageProcessing.c -o ImageProcessing -ljpeg -lm -g

run:
	./ImageProcessing sickle1.jpg thresholded.jpg equalised.jpg smoothed1.jpg smoothed2.jpg edged.jpg orientation.jpg labelled.jpg classified.jpg
	open -a Preview  *.jpg
