#!/usr/bin/expect
########################################################################
# 从远端服务器镜像同步文件到本地
########################################################################

if { $argc!=7 }  {
    send_user "Usage: script.sh remoteip remoteport remoteuser remotepwd remotedir localfile timeout\n\n"
    exit 1
}

set remoteip   [lindex $argv 0]
set remoteport [lindex $argv 1]
set remoteuser [lindex $argv 2]
set remotepwd  [lindex $argv 3]
set remotedir  [lindex $argv 4]
set localfile  [lindex $argv 5]
set timeout    [lindex $argv 6]

spawn /usr/bin/rsync -avzh --delete --append-verify -e "ssh -l$remoteuser -p$remoteport" $remoteip:$remotedir $localfile

expect {
    "password:" {
        send "$remotepwd\r"
        exp_continue
    }
    "yes/no)?" {
        send "yes\r"
        exp_continue
    }
    timeout {
        close
        break
    }
    eof {
        exit 0
    }
}
exit