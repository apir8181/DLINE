for HOST in "pc61" "pc63" "pc51" "pc52" "pc53" "pc54"
do
    echo "copy to $HOST"
    scp -r ~/distributed_LINE ${HOST}:~/ > /dev/null 2>&1
done
