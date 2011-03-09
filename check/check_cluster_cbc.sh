#!/usr/bin/env bash
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*                                                                           *
#*                  This file is part of the program and library             *
#*         SCIP --- Solving Constraint Integer Programs                      *
#*                                                                           *
#*    Copyright (C) 2002-2010 Konrad-Zuse-Zentrum                            *
#*                            fuer Informationstechnik Berlin                *
#*                                                                           *
#*  SCIP is distributed under the terms of the ZIB Academic License.         *
#*                                                                           *
#*  You should have received a copy of the ZIB Academic License              *
#*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      *
#*                                                                           *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#
# Call with "make testclustercbc"
#
# The Cluster consist of 80 nodes. These are divided into two sets of 40
# node. Each set has a different hardware configuration. Both sets can be reached
# over different queues.
# - queue "ib":  PowerEdgeTM 1950 Xeon E5420 with 2 CPUS each with 4 Cores  and 16 GB RAM
#                This gives a total of 40 * 2 * 4 = 320 cores
# - queue "gbe": PowerEdgeTM 1955 Xeon 5150 with 2 CPUS each with 2 Cores  and 8 GB RAM
#                This gives a total of 40 * 2 * 2 = 160 cores
#
# In case of time measuring you should order 1 node and 8 core (ib) or 4
# cores (gbe) depending on the used queue.  If no time is measured, change
# to PPN=1 (see below) in order to allow parallel runs on one node.  For
# more information, see "http://www.zib.de/cluster-user/view/Main/Hardware"
#
# To get the result files call "./evalcheck_cluster.sh
# results/check.$TSTNAME.$BINNMAE.$SETNAME.eval in directory check/
# This leads to result files 
#  - results/check.$TSTNAME.$BINNMAE.$SETNAME.out
#  - results/check.$TSTNAME.$BINNMAE.$SETNAME.res
#  - results/check.$TSTNAME.$BINNMAE.$SETNAME.err

# number of needed core at a certain cluster node
#  - PPN=8 means we need 8 core, therefore time measuring is possible if we use 1 node of queue "ib"
#  - PPN=4 means we need 4 core, therefore time measuring is possible if we use 1 node of queue "gbe"
#  - PPN=1 means we need one core, therefore time measuring is not possible
PPN=4
QUEUE=gbe

TSTNAME=$1
BINID=$2
SETNAME=$3
BINNAME="check/$BINID"
TIMELIMIT=$5
NODELIMIT=$6
MEMLIMIT=$7
FEASTOL=$8
MIPGAP=$9
CONTINUE=${10}
LOCK=${11}

# get current SCIP path
SCIPPATH=`pwd`

if test ! -e $SCIPPATH/results
then
    mkdir $SCIPPATH/results
fi

if test ! -e $SCIPPATH/locks
then
    mkdir $SCIPPATH/locks
fi

LOCKFILE=locks/$TSTNAME.$SETNAME.$VERSION.$LPS.lock

SETTINGS=$SCIPPATH/../settings/$SETNAME.set

# check if the settings file exists
if test $SETNAME != "default"
then
    if test ! -e $SETTINGS
    then
	echo skipping test due to not existes of the settings file $SETTINGS
	exit
    fi
fi

if test "$LOCK" = "true"
then
    if test -e $LOCKFILE
    then
	echo skipping test due to existing lock file $LOCKFILE
	exit
    fi
    date > $LOCKFILE
fi

# we add 10% to the hard time limit and additional 600 seconds in case of small time limits
# NOTE: the jobs should have a hard running time of more than 5 minutes; if not so, these
#       jobs get automatically assigned in the "exrpess" queue; this queue has only 4 CPUs
#       available 
HARDTIMELIMIT=`expr \`expr $TIMELIMIT + 600\` + \`expr $TIMELIMIT / 10\``

# we add 10% to the hard memory limit and additional 100mb to the hard memory limit
HARDMEMLIMIT=`expr \`expr $MEMLIMIT + 100\` + \`expr $MEMLIMIT / 10\``
HARDMEMLIMIT=`expr $HARDMEMLIMIT \* 1024000`

EVALFILE=$SCIPPATH/results/check.$TSTNAME.$BINID.$SETNAME.eval
echo > $EVALFILE

for i in `cat $TSTNAME.test` DONE
do
  if test "$i" = "DONE"
      then
      break
  fi

  SHORTFILENAME=`basename $i .gz`
  SHORTFILENAME=`basename $SHORTFILENAME .mps`
  SHORTFILENAME=`basename $SHORTFILENAME .lp`
  SHORTFILENAME=`basename $SHORTFILENAME .opb`

  FILENAME=$TSTNAME.$COUNT"_"$SHORTFILENAME.$BINID.$SETNAME
  BASENAME=$SCIPPATH/results/$FILENAME

  TMPFILE=$BASENAME.tmp
  SETFILE=$BASENAME.set
  
  echo $BASENAME >> $EVALFILE

  echo > $TMPFILE
  if test $FEASTOL != "default"
      then
      echo primalTolerance $FEASTOL       >> $TMPFILE
      echo integerTolerance $FEASTOL      >> $TMPFILE
  fi
  echo seconds $TIMELIMIT                 >> $TMPFILE
  if test $MIPGAP != "default"
      then
      echo ratioGap $MIPGAP               >> $TMPFILE
  fi
  echo maxNodes $NODELIMIT                >> $TMPFILE
  echo import /workbig/$i                 >> $TMPFILE
  echo ratioGap                           >> $TMPFILE
  echo allowableGap                       >> $TMPFILE
  echo seconds                            >> $TMPFILE
  echo stat                               >> $TMPFILE
  echo solve                              >> $TMPFILE
  echo quit                                >> $TMPFILE
  
  qsub -l walltime=$HARDTIMELIMIT -l mem=$HARDMEMLIMIT -l nodes=1:ppn=$PPN -N SCIP$SHORTFILENAME -v SCIPPATH=$SCIPPATH,BINNAME=$BINNAME,FILENAME=$i,BASENAME=$FILENAME -q $QUEUE -o /dev/null -e /dev/null runcluster.sh
 
  COUNT=`expr $COUNT + 1`
done
