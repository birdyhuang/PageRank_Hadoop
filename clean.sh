#########################################################################
# File Name: clean.sh
# Author: ma6174
# mail: ma6174@163.com
# Created Time: Sat 17 Oct 2015 06:52:25 AM UTC
#########################################################################
#!/bin/bash
rm log.txt
hadoop fs -rm -r output/output*

