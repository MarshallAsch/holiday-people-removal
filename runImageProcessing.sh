#/bin/bash


uploadedDir=$1
outputName=$2

echo "Running people removal"

imageProcessing/removePeople.py --no-sky "uploads/$uploadedDir"

status=$?
if [ $status -eq 0 ]
then
    echo "success"
    cp imageProcessing/final_noSky.png "public/generated/$outputName"
fi

echo "done"

exit $status