for f in *.java
do
cat $f | sed 's/\r/\n/g' > $f.out
mv $f.out $f
done

