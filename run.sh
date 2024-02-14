# The goal is to check if a new file is added to the input folder and if so, run the script main.py

while true
do
    if [ ! -d "./input" ]; then
        echo "Input folder does not exist"
        echo "Creating input folder"
        mkdir input
    fi
    if [ "$(ls -A ./input)" ]; then
        if [ ! -d "./log" ]; then
            mkdir log
        fi
        echo "Input folder is not empty"
        echo "Running main.py"
        python3 main.py
        echo "Done running main.py"
    else
        echo "Input folder is empty"
    fi
    sleep 30
done