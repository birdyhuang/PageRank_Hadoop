#########################################################################
# File Name: run.sh
# Author: ma6174
# mail: ma6174@163.com
# Created Time: Sat 17 Oct 2015 06:52:44 AM UTC
#########################################################################
#!/bin/bash
gradle jar
hadoop jar build/libs/Gradle_PageRank.jar input/small.txt output/ >> log.txt
