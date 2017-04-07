#!/bin/sh

git describe --always > VERSION

if [ $? -ne 0 ]
then
    echo "1.0" > VERSION
fi

