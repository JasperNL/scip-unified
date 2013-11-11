#!/usr/bin/env bash
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*                                                                           *
#*                  This file is part of the program and library             *
#*         SCIP --- Solving Constraint Integer Programs                      *
#*                                                                           *
#*    Copyright (C) 2002-2013 Konrad-Zuse-Zentrum                            *
#*                            fuer Informationstechnik Berlin                *
#*                                                                           *
#*  SCIP is distributed under the terms of the ZIB Academic License.         *
#*                                                                           *
#*  You should have received a copy of the ZIB Academic License              *
#*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      *
#*                                                                           *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
TSTNAME=$1
BINNAME=$2
SETNAME=$3
BINID=$4
TIMELIMIT=$5
NODELIMIT=$6
MEMLIMIT=$7
THREADS=$8
FEASTOL=$9
DISPFREQ=${10}
CONTINUE=${11}
LOCK=${12}
VERSION=${13}
LPS=${14}
VALGRIND=${15}

# check if all variables defined (by checking the last one)
if test -z $VALGRIND
then
    echo Skipping test since not all variables are define
    echo "TSTNAME       = $TSTNAME"
    echo "BINNAME       = $BINNAME"
    echo "SETNAME       = $SETNAME"
    echo "BINID         = $BINID"
    echo "TIMELIMIT     = $TIMELIMIT"
    echo "NODELIMIT     = $NODELIMIT"
    echo "MEMLIMIT      = $MEMLIMIT"
    echo "THREADS       = $THREADS"
    echo "FEASTOL       = $FEASTOL"
    echo "DISPFREQ      = $DISPFREQ"
    echo "CONTINUE      = $CONTINUE"
    echo "LOCK          = $LOCK"
    echo "VERSION       = $VERSION"
    echo "LPS           = $LPS"
    echo "VALGRIND      = $VALGRIND"
    exit 1;
fi

SETDIR=../settings

if test ! -e results
then
    mkdir results
fi
if test ! -e locks
then
    mkdir locks
fi

# set this to 1 if you want the scripts to (try to) pass a best known primal bound (from .solu file) to the GAMS solver
SETCUTOFF=0

LOCKFILE=locks/$TSTNAME.$SETNAME.$VERSION.$LPS.lock
RUNFILE=locks/$TSTNAME.$SETNAME.$VERSION.$LPS.run.$BINID
DONEFILE=locks/$TSTNAME.$SETNAME.$VERSION.$LPS.done

OUTFILE=results/check.$TSTNAME.$BINID.$SETNAME.out
ERRFILE=results/check.$TSTNAME.$BINID.$SETNAME.err
RESFILE=results/check.$TSTNAME.$BINID.$SETNAME.res
TEXFILE=results/check.$TSTNAME.$BINID.$SETNAME.tex
TMPFILE=results/check.$TSTNAME.$BINID.$SETNAME.tmp
SETFILE=results/check.$TSTNAME.$BINID.$SETNAME.set

# if cutoff should be passed, check for solu file
if test $SETCUTOFF = 1 ; then
  if test -e testset/$TSTNAME.solu ; then
    SOLUFILE=testset/$TSTNAME.solu
  elif test -e testset/all.solu ; then
    SOLUFILE=testset/all.solu
  else
    echo "Warning: SETCUTOFF=1 set, but no .solu file (testset/$TSTNAME.solu or testset/all.solu) available"
    SETCUTOFF=0
  fi
fi

SETTINGS=$SETDIR/$SETNAME.set

if test "$LOCK" = "true"
then
    if test -e $DONEFILE
    then
        echo skipping test due to existing done file $DONEFILE
        exit
    fi
    if test -e $LOCKFILE
    then
        if test -e $RUNFILE
        then
            echo continuing aborted run with run file $RUNFILE
        else
            echo skipping test due to existing lock file $LOCKFILE
            exit
        fi
    fi
    date > $LOCKFILE
    date > $RUNFILE
fi

if test ! -e $OUTFILE
then
    CONTINUE=false
fi

if test "$CONTINUE" = "true"
then
    MVORCP=cp
else
    MVORCP=mv
fi

DATEINT=`date +"%s"`
if test -e $OUTFILE
then
    $MVORCP $OUTFILE $OUTFILE.old-$DATEINT
fi
if test -e $ERRFILE
then
    $MVORCP $ERRFILE $ERRFILE.old-$DATEINT
fi

if test "$CONTINUE" = "true"
then
    LASTPROB=`awk -f getlastprob.awk $OUTFILE`
    echo Continuing benchmark. Last solved instance: $LASTPROB
    echo "" >> $OUTFILE
    echo "----- Continuing from here. Last solved: $LASTPROB -----" >> $OUTFILE
    echo "" >> $OUTFILE
else
    LASTPROB=""
fi

uname -a >>$OUTFILE
uname -a >>$ERRFILE
date >>$OUTFILE
date >>$ERRFILE

# we add 10% to the hard time limit and additional 10 seconds in case of small time limits
HARDTIMELIMIT=`expr \`expr $TIMELIMIT + 10\` + \`expr $TIMELIMIT / 10\``

# we add 10% to the hard memory limit and additional 100mb to the hard memory limit
HARDMEMLIMIT=`expr \`expr $MEMLIMIT + 1000\` + \`expr $MEMLIMIT / 10\``
HARDMEMLIMIT=`expr $HARDMEMLIMIT \* 1024`

echo "hard time limit: $HARDTIMELIMIT s" >>$OUTFILE
echo "hard mem limit: $HARDMEMLIMIT k" >>$OUTFILE

# check if the test run should be processed in the valgrind environment
if test "$VALGRIND" = "true"
then
    VALGRINDCMD="valgrind --log-fd=1 --leak-check=full"
else
    VALGRINDCMD=""
fi

for i in `cat testset/$TSTNAME.test` DONE
do
    if test "$i" = "DONE"
    then
        date > $DONEFILE
        break
    fi

    if test "$LASTPROB" = ""
    then
        LASTPROB=""
        if test -f $i
        then
	    SHORTFILENAME=`basename $i .gz`
	    SHORTFILENAME=`basename $SHORTFILENAME .mps`
	    SHORTFILENAME=`basename $SHORTFILENAME .lp`
	    SHORTFILENAME=`basename $SHORTFILENAME .opb`
	    SHORTFILENAME=`basename $SHORTFILENAME .gms`
	    SHORTFILENAME=`basename $SHORTFILENAME .pip`
	    SHORTFILENAME=`basename $SHORTFILENAME .zpl`
	    SHORTFILENAME=`basename $SHORTFILENAME .cip`
	    SHORTFILENAME=`basename $SHORTFILENAME .fzn`
	    SHORTFILENAME=`basename $SHORTFILENAME .osil`
	    SHORTFILENAME=`basename $SHORTFILENAME .wbo`
	    SHORTFILENAME=`basename $SHORTFILENAME .cnf`

	    if test $SETCUTOFF = 1
	    then
		export CUTOFF=`grep "$SHORTFILENAME " $SOLUFILE | grep -v =feas= | grep -v =inf= | tail -n 1 | awk '{print $3}'`
		echo CUTOFF:  $CUTOFF
	    fi

            echo @01 $i ===========
            echo @01 $i ===========                >> $ERRFILE
            echo > $TMPFILE
            if test "$SETNAME" != "default"
            then
                echo set load $SETTINGS            >>  $TMPFILE
            fi
            if test "$FEASTOL" != "default"
            then
                echo set numerics feastol $FEASTOL >> $TMPFILE
            fi
            echo set limits time $TIMELIMIT        >> $TMPFILE
            echo set limits nodes $NODELIMIT       >> $TMPFILE
            echo set limits memory $MEMLIMIT       >> $TMPFILE
            echo set lp advanced threads $THREADS  >> $TMPFILE
            echo set timing clocktype 1            >> $TMPFILE
            echo set display freq $DISPFREQ        >> $TMPFILE
            echo set memory savefac 1.0            >> $TMPFILE # avoid switching to dfs - better abort with memory error
            if test "$LPS" = "none"      
            then
                echo set lp solvefreq -1           >> $TMPFILE # avoid solving LPs in case of LPS=none
            fi
            echo set save $SETFILE                 >> $TMPFILE
            echo read $i                           >> $TMPFILE
	    if test $SETCUTOFF = 1
	    then
		if test $CUTOFF != ""
		then
		    echo set limits objective $CUTOFF      >> $TMPFILE
		fi
		echo set heur emph off                 >> $TMPFILE
		echo set sepa emph off                 >> $TMPFILE
	    fi
#            echo write genproblem cipreadparsetest.cip >> $TMPFILE
#            echo read cipreadparsetest.cip         >> $TMPFILE
            echo optimize                          >> $TMPFILE
            echo display statistics                >> $TMPFILE
#           echo display solution                  >> $TMPFILE
            echo checksol                          >> $TMPFILE
            echo quit                              >> $TMPFILE
            echo -----------------------------
            date
            date >>$ERRFILE
            echo -----------------------------
            date +"@03 %s"
            bash -c " ulimit -t $HARDTIMELIMIT s; ulimit -v $HARDMEMLIMIT k; ulimit -f 200000; $VALGRINDCMD ../$BINNAME < $TMPFILE" 2>>$ERRFILE
            date +"@04 %s"
            echo -----------------------------
            date
            date >>$ERRFILE
            echo -----------------------------
            echo
            echo =ready=
        else
            echo @02 FILE NOT FOUND: $i ===========
            echo @02 FILE NOT FOUND: $i =========== >>$ERRFILE
        fi
    else
        echo skipping $i
        if test "$LASTPROB" = "$i"
        then
            LASTPROB=""
        fi
    fi
done | tee -a $OUTFILE

rm -f $TMPFILE
rm -f cipreadparsetest.cip

date >>$OUTFILE
date >>$ERRFILE

if test -e $DONEFILE
then
    ./evalcheck.sh $OUTFILE
    
    if test "$LOCK" = "true"
    then
        rm -f $RUNFILE
    fi
fi
