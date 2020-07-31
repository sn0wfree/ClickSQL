#!/usr/bin/env bash

branch=`git symbolic-ref --short HEAD`

git add *
date_string=`date`
echo $date_string
msg="git auto commit at ${date_string}"
git commit -m"$msg"
git push origin $branch