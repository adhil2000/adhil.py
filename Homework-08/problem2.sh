#!/usr/bin/env bash

OUTfile=${OUTFILE}.out
echo $OUTfile
ERRfile=${OUTFILE}.err
echo $ERRfile
./cmd1 < $INFILE | ./cmd3 1> $OUTfile 2> $ERRfile
