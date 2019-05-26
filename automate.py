from subprocess import call
call("git pull origin master",shell=True)
call("python twitter_post.py",shell=True)
