DATAPATH='input_videos'

for file in `ls $DATAPATH/*.avi`
do
dirname=${file%.*}
echo $dirname
mkdir $dirname
# ffmpeg -i $file $dirname/%08d.jpg
done