# $Id: check_cplex.sh,v 1.3 2003/12/23 12:13:06 bzfpfend Exp $
CPLEXBIN=cplex
TSTNAME=$1
BINNAME=$CPLEXBIN.$2
TIMELIMIT=$3
NODELIMIT=$4
MEMLIMIT=$5

OUTFILE=results/check.$TSTNAME.$BINNAME.out
ERRFILE=results/check.$TSTNAME.$BINNAME.err
TEXFILE=results/check.$TSTNAME.$BINNAME.tex
TMPFILE=results/check.$TSTNAME.$BINNAME.tmp

uname -a >$OUTFILE
uname -a >$ERRFILE
date >>$OUTFILE
date >>$ERRFILE

for i in `cat $TSTNAME.test`
do
    if [ -f $i ]
    then
	echo @01 $i ===========
	echo @01 $i ===========                 >> $ERRFILE
	echo read $i                             > $TMPFILE
	echo set timelimit $TIMELIMIT           >> $TMPFILE
	echo set clocktype 1                    >> $TMPFILE
	echo set mip interval 10000             >> $TMPFILE
	echo set mip tolerances absmipgap 1e-10 >> $TMPFILE
	echo set mip tolerances mipgap 0.0      >> $TMPFILE
	echo set mip limits nodes $NODELIMIT    >> $TMPFILE
	echo set mip limits treememory $MEMLIMIT >> $TMPFILE
	echo optimize                           >> $TMPFILE
	echo quit                               >> $TMPFILE
	$CPLEXBIN < $TMPFILE 2>>$ERRFILE
	echo =ready=
    else
	echo @02 FILE NOT FOUND: $i ===========
	echo @02 FILE NOT FOUND: $i =========== >>$ERRFILE
    fi
done | tee -a $OUTFILE

rm $TMPFILE

date >>$OUTFILE
date >>$ERRFILE

gawk -f check_cplex.awk $OUTFILE > $TEXFILE
