rsync -av --progress -e "ssh -p 20022" --ignore-existing taerim@163.152.162.50:/home/taerim/AMP-STMR/logs/ logs50/ &
rsync -av --progress -e "ssh -p 20022" --ignore-existing taerim@163.152.162.63:/home/taerim/AMP-STMR/logs/ logs63/&
rsync -av --progress -e "ssh -p 20022" --ignore-existing taerim@163.152.162.64:/home/taerim/AMP-STMR/logs/ logs64/ &
wait