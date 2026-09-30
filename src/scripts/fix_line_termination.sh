
if [ "$#" -ne 1 ]; then
    echo "usage: fix_line_termination.sh <filename>"
    exit 1
fi

cat $1 | sed 's/\r/\n/g' > $1.out
mv $1.out $1


