#!/usr/bin/awk -f
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
#*                                                                           *
#*                  This file is part of the program and library             *
#*         SCIP --- Solving Constraint Integer Programs                      *
#*                                                                           *
#*    Copyright (C) 2002-2011 Konrad-Zuse-Zentrum                            *
#*                            fuer Informationstechnik Berlin                *
#*                                                                           *
#*  SCIP is distributed under the terms of the ZIB Academic License.         *
#*                                                                           *
#*  You should have received a copy of the ZIB Academic License              *
#*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      *
#*                                                                           *
#* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# $Id: check_gams.awk,v 1.3 2011/01/28 16:17:33 bzfviger Exp $
#
#@file    check_gams.awk
#@brief   GAMS Tracefile Check Report Generator
#@author  Robert Waniek
#@author  Stefan Vigerske
#@author  check.awk (copied large portions from there)
#
function max(x,y)
{
    return (x) > (y) ? (x) : (y);
}
function min(x,y)
{
   return (x) < (y) ? (x) : (y);
}
function abs(a)
{
  if (a>0) return a;
  else return -a;
}
function isEQ(a, b)
{
  scale = abs(a);
  if ( scale >= 1e+10 )
    scale = abs(b);
  if ( scale < 0.0001 )
    scale = 1;
  return abs(a-b) < 0.0002 * scale;
}

BEGIN  { 
   timegeomshift = 10.0;
   nodegeomshift = 100.0;
   pavshift = 1.0;
   onlyinsolufile = 0;       # should only instances be reported that are included in the .solu file?
   useshortnames = 1;        # should problem name be truncated to fit into column?
   writesolufile = 0;        # should a solution file be created from the results
   NEWSOLUFILE = "new_solufile.solu";
   infty = 1e+20;

   if (solver == "") solver="SCIP";
   if (penaltytime == "") penaltytime=3600.0;

   printf("------------------+------+--- Original --+-- Presolved --+----------------+----------------+------+--------+-------+-------+-------\n");
   printf("Name              | Type | Conss |  Vars | Conss |  Vars |   Dual Bound   |  Primal Bound  | Gap%% |  Iters | Nodes |  Time |       \n");
   printf("------------------+------+-------+-------+---------------+----------------+----------------+------+--------+-------+-------+-------\n");

   nprobs = 0;
   sbab = 0;
   ssim = 0;
   stottime = 0.0;
   nodegeom = 0.0;
   timegeom = 0.0;
   shiftednodegeom = nodegeomshift;
   shiftedtimegeom = timegeomshift;
   timeouttime = 0.0;
   timeouts = 0;
   failtime = 0.0;
   fail = 0;
   pass = 0;
   timelimit = 0;
   settings = "default";
}
/=opt=/  { solstatus[$2] = "opt"; sol[$2] = $3; }   # get optimum
/=inf=/  { solstatus[$2] = "inf"; }                 # problem infeasible (no feasible solution exists)
/=best=/ { solstatus[$2] = "best"; sol[$2] = $3; }  # get best known solution value
/=feas=/ { solstatus[$2] = "feas"; }                # no feasible solution known
/=unkn=/ { solstatus[$2] = "unkn"; }                # no feasible solution known
/^\*/    { FS="," }  # a start at the beginnnig of a line we take as start of tracefile, so change FS to ','
/^\* SOLVER/     { solver=$2; }
/^\* TIMELIMIT/  { timelimit=$2; }
/^\* SETTINGS/   { settings=$2; }
/^\*/  { next; } # skip other comments and invalid problems
/^ *$/ { next; } # skip empty lines

#These need to coincide with those in in check_gams.sh
#TODO make this more flexible (see readtrace.awk from ptools)
#01 InputFileName
#02 ModelType
#03 SolverName
#04 OptionFile
#05 Direction
#06 NumberOfEquations
#07 NumberOfVariables
#08 NumberOfDiscreteVariables
#09 NumberOfNonZeros
#10 NumberOfNonlinearNonZeros
#11 ModelStatus
#12 SolverStatus
#13 ObjectiveValue
#14 ObjectiveValueEstimate
#15 SolverTime
#16 ETSolver
#17 NumberOfIterations
#18 NumberOfNodes

/.*/ {
  if ( $3 == solver )
  {
    model[nprobs] = $1;
    type[nprobs] = $2;
    maxobj[nprobs] = ( $5 == 1 ? 1 : 0 );
    cons[nprobs] = $6;
    vars[nprobs] = $7;
    modstat[nprobs] = $11;
    solstat[nprobs] = $12;
    dualbnd[nprobs] = $14;
    primalbnd[nprobs] = $13;
    time[nprobs] = $16;
    iters[nprobs] = $17;
    nodes[nprobs] = $18;
    nprobs++;
  }
}

END {
  for (m = 0; m < nprobs; m++)
  {
     prob = model[m];
     if( useshortnames && length(prob) > 18 )
       shortprob = substr(prob, length(prob)-17, 18);
     else
       shortprob = prob;

     if (primalbnd[m] == "NA")
       primalbnd[m] = ( maxobj[m] ? -infty : +infty );
       
     # do not trust primal bound in trace file if model status indicates that no feasible solution was found
     if( modstat[m] != 1 && modstat[m] != 2 && modstat[m] != 3 && modstat[m] != 7 && modstat[m] != 8 )
       primalbnd[m] = ( maxobj[m] ? -infty : +infty );

     # if dual bound is not given, but solver claimed model status "optimal", then we set dual bound to primal bound
     if (dualbnd[m] == "NA")
       dualbnd[m] = ( modstat[m] == 1 ? primalbnd[m] : ( maxobj[m] ? +infty : -infty ) );

     db = dualbnd[m];
     pb = primalbnd[m];
     
     # we consider everything from solver status 4 on as unusal interrupt, i.e., abort
     aborted = 0;
     if( solstat[m] >= 4 )
       aborted = 1;

     # TODO consider gaplimit
     gapreached = 0
     
     timeout = 0

     if( !onlyinsolufile || solstatus[prob] != "" )  {

       #avoid problems when comparing floats and integer (make everything float)
       temp = pb;
       pb = 1.0*temp;
       temp = db;
       db = 1.0*temp;
      
       optimal = 0;
       markersym = "\\g";
       if( abs(pb - db) < 1e-06 && pb < infty ) {
         gap = 0.0;
         optimal = 1;
         markersym = "  ";
       }
       else if( abs(db) < 1e-06 )
         gap = -1.0;
       else if( pb*db < 0.0 )
         gap = -1.0;
       else if( abs(db) >= +infty )
         gap = -1.0;
       else if( abs(pb) >= +infty )
         gap = -1.0;
       else
         gap = 100.0*abs((pb-db)/db);
       if( gap < 0.0 )
         gapstr = "  --  ";
       else if( gap < 1e+04 )
         gapstr = sprintf("%6.1f", gap);
       else
         gapstr = " Large";
      
       probtype = type[m];

       if( time[m] > timelimit && timelimit > 0.0 ) {
         timeout = 1;
         tottime = time[m];
       }
       else if( gapreached )
         timeout = 0;
       if( tottime == 0.0 )
         tottime = timelimit;
       if( timelimit > 0.0 )
         tottime = min(tottime, timelimit);

       simplex = iters[m];
       stottime += tottime;
       sbab += nodes[m];
       ssim += simplex;

       nodegeom = nodegeom * max(nodes[m], 1.0)^(1.0/nprobs);
       timegeom = timegeom * max(tottime, 1.0)^(1.0/nprobs);

       shiftednodegeom = shiftednodegeom * max(nodes[m]+nodegeomshift, 1.0)^(1.0/nprobs);
       shiftedtimegeom = shiftedtimegeom * max(tottime+timegeomshift, 1.0)^(1.0/nprobs);

       status = "";
       if( readerror ) {
         status = "readerror";
         failtime += tottime;
         fail++;
       }
       else if( aborted ) {
         status = "abort";
         failtime += tottime;
         fail++;
       }
       else if( solstatus[prob] == "opt" ) {
         reltol = 1e-5 * max(abs(pb),1.0);
         abstol = 1e-4;

         if( ( !maxobj[m] && (db-sol[prob] > reltol || sol[prob]-pb > reltol) ) || ( maxobj[m] && (sol[prob]-db > reltol || pb-sol[prob] > reltol) ) ) {
            status = "fail";
            failtime += tottime;
            fail++;
         }
         else {
           if( timeout || gapreached ) {
             if( timeout )
                status = "timeout";
             else if( gapreached )
                status = "gaplimit";
             timeouttime += tottime;
             timeouts++;
           }
           else {
             if( (abs(pb - db) <= max(abstol, reltol)) && abs(pb - sol[prob]) <= reltol ) {
               status = "ok";
               pass++;
             }
             else {
               status = "fail";
               failtime += tottime;
               fail++;
             }
           }
         }
       }
       else if( solstatus[prob] == "best" ) {
         reltol = 1e-5 * max(abs(pb),1.0);
         abstol = 1e-4;

         if( ( !maxobj[m] && db-sol[prob] > reltol) || ( maxobj[m] && sol[prob]-db > reltol) ) {
           status = "fail";
           failtime += tottime;
           fail++;
         }
         else {
           if( timeout || gapreached ) {
             if( (!maxobj[m] && sol[prob]-pb > reltol) || (maxobj[m] && pb-sol[prob] > reltol) ) {
               status = "better";
               timeouttime += tottime;
               timeouts++;
             }
             else {
               if( timeout )
                 status = "timeout";
               else if( gapreached )
                 status = "gaplimit";
               timeouttime += tottime;
               timeouts++;
             }
           }
           else {
             if( abs(pb - db) <= max(abstol, reltol) ) {
               status = "solved";
               pass++;
             }
             else {
               status = "fail";
               failtime += tottime;
               fail++;
             }
           }
         }
       }
       else if( solstatus[prob] == "unkn" ) {
         reltol = 1e-5 * max(abs(pb),1.0);
         abstol = 1e-4;
         
         if( abs(pb - db) <= max(abstol, reltol) ) {
           status = "solved not verified";
           pass++;
         }
         else {
           if( abs(pb) < infty ) {
             status = "better";
             timeouttime += tottime;
             timeouts++;
           }
           else {
             if( timeout || gapreached ) {
               if( timeout )
                 status = "timeout";
               else if( gapreached )
                 status = "gaplimit";
               timeouttime += tottime;
               timeouts++;
             }
             else 
               status = "unknown";
           }
         }
       }
       else if( solstatus[prob] == "inf" ) {
         if( !feasible ) {
           if( timeout ) {
             status = "timeout";
             timeouttime += tottime;
             timeouts++;
           }
           else {
             status = "ok";
             pass++;
           }
         }
         else {
           status = "fail";
           failtime += tottime;
           fail++;
         }
       }
       else if( solstatus[prob] == "feas" ) {
         if( feasible ) {
           if( timeout ) {
             status = "timeout";
             timeouttime += tottime;
             timeouts++;
           }
           else {
             status = "ok";
             pass++;
           }
         }
         else {
           status = "fail";
           failtime += tottime;
           fail++;
         }
       }
       else {
         reltol = 1e-5 * max(abs(pb),1.0);
         abstol = 1e-4;

         if( abs(pb - db) < max(abstol,reltol) ) {
           status = "solved not verified";
           pass++;
         }
         else {
           if( timeout || gapreached ) {
             if( timeout )
               status = "timeout";
             else if( gapreached )
               status = "gaplimit";
             timeouttime += tottime;
             timeouts++;
           }
           else
             status = "unknown";
         }
       }

       if( writesolufile ) {
         if( pb == +infty && db == +infty )
           printf("=inf= %-18s\n",prob)>NEWSOLUFILE;
         else if( pb == db )
           printf("=opt= %-18s %16.9g\n",prob,pb)>NEWSOLUFILE;
         else if( pb < +infty )
           printf("=best= %-18s %16.9g\n",prob,pb)>NEWSOLUFILE;
         else
           printf("=unkn= %-18s\n",prob)>NEWSOLUFILE;
         #=feas= cannot happen since the problem is reported with an objective value
       }

       #write output to both the tex file and the console depending on whether printsoltimes is activated or not
       printf("%-19s & %6d & %6d & %16.9g & %16.9g & %6s &%s%8d &%s%7.1f",
              pprob, cons[m], vars[m], db, pb, gapstr, markersym, nodes[m], markersym, time[m])  >TEXFILE;
       printf("\\\\\n") > TEXFILE;

       printf("%-19s %-5s %7d %7d      ??      ?? %16.9g %16.9g %6s %8d %7d %7.1f %s (%2d - %2d)\n",
              shortprob, probtype, cons[m], vars[m], db, pb, gapstr, iters[m], nodes[m], time[m], status, modstat[m], solstat[m]);

       #PAVER output: see http://www.gamsworld.org/performance/paver/pprocess_submit.htm
       if( solstatus[prob] == "opt" || solstatus[prob] == "feas" )
         modelstat = 1;
       else if( solstatus[prob] == "inf" )
         modelstat = 1;
       else if( solstatus[prob] == "best" )
         modelstat = 8;
       else
         modelstat = 1;
       if( status == "ok" || status == "unknown" )
         solverstat = 1;
       else if( status == "timeout" )
         solverstat = 3;
       else
         solverstat = 10;
       pavprob = prob;
       if( length(pavprob) > 25 )
         pavprob = substr(pavprob, length(pavprob)-24,25);
       printf("%s,MIP,%s_%s,0,%d,%d,%g,%g\n", pavprob, solver, settings, modelstat, solverstat, pb, tottime+pavshift) > PAVFILE;
     }
   }
   printf("------------------+------+-------+-------+-------+-------+----------------+----------------+------+--------+-------+-------+-------\n");
   printf("\n");
   printf("------------------------------[Nodes]---------------[Time]------\n");
   printf("  Cnt  Pass  Time  Fail  total(k)     geom.     total     geom. \n");
   printf("----------------------------------------------------------------\n");
   printf("%5d %5d %5d %5d %9d %9.1f %9.1f %9.1f\n",
     nprobs, pass, timeouts, fail, sbab / 1000, nodegeom, stottime, timegeom);
   printf(" shifted geom. [%5d/%5.1f]      %9.1f           %9.1f\n",
     nodegeomshift, timegeomshift, shiftednodegeom, shiftedtimegeom);
   printf("----------------------------------------------------------------\n");
   printf("@01 %s\n", solver);
   printf("\n");
}
