BRANCH=`git rev-parse --abbrev-ref HEAD`
echo $BRANCH > task_name
scp -r ./* root@172.16.0.227:/home/zmp/projects/undnow