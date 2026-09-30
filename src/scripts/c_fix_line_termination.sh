for f in *.c
do
cat $f | sed 's/\r/\n/g' > $f.out
mv $f.out $f
done

