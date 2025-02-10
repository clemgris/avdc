#!/bin/bash

if [ "$1" = "lorel" ]
then
    list_text=(
        "close drawer"
        "open drawer"
        "turn faucet left"
        "turn faucet right"
        "move white mug up"
        "move black mug up"
        "move white mug down"
        "move black mug down"
    )

    # Iterate over each instruction
    for text in "${list_text[@]}"; do
        echo "Generating trajectories for: $text"
        python train/train_lorel.py -m inference -c 20 -p /home/grislain/AVDC/init_img/lorel_init_img.png -t "$text" -num 10 -g 3
    done
    echo "Done generating trajectories for Lorel."
elif [ "$1" = "calvin" ]
then
    list_text=(
            "open drawer"
            "turn on led"
            "turn on lightbulb"
            "move slider left"
            "lift pink block table"
            "rotate pink block right"
            "lift blue block slider"
        )

    # Iterate over each instruction
    for text in "${list_text[@]}"; do
        echo "Generating trajectories for: $text"
        python train/train_calvin.py -m inference -c 12 -p /home/grislain/AVDC/init_img/calvin_init_img.png -t "$text" -num 10 -g 3
    done
    echo "Done generating trajectories for Calvin."
else
    echo "Invalid argument. Please specify either 'lorel' or 'calvin'."
fi