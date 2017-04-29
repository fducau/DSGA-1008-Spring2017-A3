#!/bin/bash

for FILE in ./data/img_align_celeba/*jpg
do
	echo $FILE
	convert $FILE -filter Catrom -resize 25% ./data/resized/$FILE
done



 #convert ./img_align_celeba/00000*.jpg -filter Catrom -resize 25% ./img_align_celeba_resized_25/out%d.jpg