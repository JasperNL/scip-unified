/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2010 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#pragma ident "@(#) $Id: heur_mutation.c,v 1.11.2.2 2010/03/22 16:05:21 bzfwolte Exp $"

/**@file   heur_mutation.c
 * @ingroup PRIMALHEURISTICS
 * @brief  mutation primal heuristic
 * @author Timo Berthold
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include "scip/scip.h"
#include "scip/scipdefplugins.h"
#include "scip/cons_linear.h"
#include "scip/heur_mutation.h"
#include "scip/pub_misc.h"

#define HEUR_NAME             "mutation"
#define HEUR_DESC             "mutation heuristic randomly fixing variables"
#define HEUR_DISPCHAR         'M'
#define HEUR_PRIORITY         -1103000
#define HEUR_FREQ             -1
#define HEUR_FREQOFS          8
#define HEUR_MAXDEPTH         -1
#define HEUR_TIMING           SCIP_HEURTIMING_AFTERNODE

#define DEFAULT_NODESOFS      500           /* number of nodes added to the contingent of the total nodes          */
#define DEFAULT_MAXNODES      5000          /* maximum number of nodes to regard in the subproblem                 */
#define DEFAULT_MINIMPROVE    0.01          /* factor by which Mutation should at least improve the incumbent      */
#define DEFAULT_MINNODES      500           /* minimum number of nodes to regard in the subproblem                 */
#define DEFAULT_MINFIXINGRATE 0.8           /* minimum percentage of integer variables that have to be fixed       */
#define DEFAULT_NODESQUOT     0.1           /* subproblem nodes in relation to nodes of the original problem       */
#define DEFAULT_NWAITINGNODES 200           /* number of nodes without incumbent change that heuristic should wait */



/*
 * Data structures
 */

/** primal heuristic data */
struct SCIP_HeurData
{
   int                   nodesofs;          /**< number of nodes added to the contingent of the total nodes          */
   int                   maxnodes;          /**< maximum number of nodes to regard in the subproblem                 */
   int                   minnodes;          /**< minimum number of nodes to regard in the subproblem                 */
   SCIP_Real             minfixingrate;     /**< minimum percentage of integer variables that have to be fixed       */
   int                   nwaitingnodes;     /**< number of nodes without incumbent change that heuristic should wait */
   SCIP_Real             minimprove;        /**< factor by which Mutation should at least improve the incumbent      */
   SCIP_Longint          usednodes;         /**< nodes already used by Mutation in earlier calls                     */
   SCIP_Real             nodesquot;         /**< subproblem nodes in relation to nodes of the original problem       */
   unsigned int          randseed;          /**< seed value for random number generator                              */
};



/*
 * Local methods
 */

/** creates a subproblem for subscip by fixing a number of variables */
static
SCIP_RETCODE createSubproblem(
   SCIP*                 scip,               /**< original SCIP data structure                                  */
   SCIP*                 subscip,            /**< SCIP data structure for the subproblem                        */
   SCIP_VAR**            subvars,            /**< the variables of the subproblem                               */
   SCIP_Real             minfixingrate,      /**< percentage of integer variables that have to be fixed         */
   unsigned int*         randseed
   )
{
   SCIP_VAR** vars;                          /* original scip variables                    */
   SCIP_ROW** rows;                          /* original scip rows                         */
   SCIP_SOL* sol;                            /* pool of solutions                          */
   SCIP_Bool* marked;                        /* array of markers, which variables to fixed */
   SCIP_Bool fixingmarker;                   /* which flag should label a fixed variable?  */

   int nrows;
   int nvars; 
   int nbinvars;
   int nintvars;
   int i;
   int j;
   int nmarkers;

   char consname[SCIP_MAXSTRLEN];  

   /* get required data of the original problem */
   SCIP_CALL( SCIPgetVarsData(scip, &vars, &nvars, &nbinvars, &nintvars, NULL, NULL) );
   sol = SCIPgetBestSol(scip);
   assert( sol != NULL );


   SCIP_CALL( SCIPallocBufferArray(scip, &marked, nbinvars+nintvars) );

   /* get name of the original problem and add the string "_mutationsub" */
   (void) SCIPsnprintf(consname, SCIP_MAXSTRLEN, "%s_mutationsub", SCIPgetProbName(scip));

   /* create the subproblem */
   SCIP_CALL( SCIPcreateProb(subscip, consname, NULL, NULL, NULL, NULL, NULL, NULL) );

   if( minfixingrate > 0.5 )
   {
      nmarkers = nbinvars + nintvars - (int) SCIPfloor(scip, minfixingrate*(nbinvars+nintvars));
      fixingmarker = FALSE;
   }
   else
   {
      nmarkers = (int) SCIPceil(scip, minfixingrate*(nbinvars+nintvars));
      fixingmarker = TRUE;
   }
   assert( 0 <= nmarkers && nmarkers <=  SCIPceil(scip,(nbinvars+nintvars)/2.0 ) );
   
   j = 0;
   BMSclearMemoryArray(marked, nbinvars+nintvars);
   while( j < nmarkers )
   {
      do
      {
         i = SCIPgetRandomInt(0, nbinvars+nintvars-1, randseed);
      }
      while( marked[i] ); 
      marked[i] = TRUE;
      j++;
   }
   assert( j == nmarkers );

   /* create variables of the subproblem and fix them to solution's value*/  
   for( i = 0; i < nbinvars + nintvars; i++ )
   { 
      if( marked[i] == fixingmarker )
      {
         SCIP_Real solval;
         solval = SCIPgetSolVal(scip, sol, vars[i]);
         SCIP_CALL( SCIPcreateVar(subscip, &subvars[i], SCIPvarGetName(vars[i]), solval,
               solval, SCIPvarGetObj(vars[i]), SCIPvarGetType(vars[i]),
               SCIPvarIsInitial(vars[i]), SCIPvarIsRemovable(vars[i]), NULL, NULL, NULL, NULL) );
      }
      else
      {
         SCIP_CALL( SCIPcreateVar(subscip, &subvars[i], SCIPvarGetName(vars[i]), SCIPvarGetLbGlobal(vars[i]),
               SCIPvarGetUbGlobal(vars[i]), SCIPvarGetObj(vars[i]), SCIPvarGetType(vars[i]),
               SCIPvarIsInitial(vars[i]), SCIPvarIsRemovable(vars[i]), NULL, NULL, NULL, NULL) );
      }
      SCIP_CALL( SCIPaddVar(subscip, subvars[i]) );
   }
   
   /* create continuous variables of the subproblem and leave them unfixed */  
   for( i = nbinvars + nintvars; i < nvars; i++ )
   {
      SCIP_CALL( SCIPcreateVar(subscip, &subvars[i], SCIPvarGetName(vars[i]), SCIPvarGetLbGlobal(vars[i]),
            SCIPvarGetUbGlobal(vars[i]), SCIPvarGetObj(vars[i]), SCIPvarGetType(vars[i]),
            SCIPvarIsInitial(vars[i]), SCIPvarIsRemovable(vars[i]), NULL, NULL, NULL, NULL) );
      SCIP_CALL( SCIPaddVar(subscip, subvars[i]) ); 
   }
      
   /* get the rows and their number */
   SCIP_CALL( SCIPgetLPRowsData(scip, &rows, &nrows) ); 
   
   /* copy all rows to linear constraints */
   for( i = 0; i < nrows; i++ )
   {
      SCIP_CONS* cons;
      SCIP_VAR** consvars;
      SCIP_COL** cols;
      SCIP_Real constant;
      SCIP_Real lhs;
      SCIP_Real rhs;
      SCIP_Real* vals;
      int nnonz;
          
      /* ignore rows that are only locally valid */
      if( SCIProwIsLocal(rows[i]) )
         continue;
      
      /* get the row's data */
      constant = SCIProwGetConstant(rows[i]);
      lhs = SCIProwGetLhs(rows[i]) - constant;
      rhs = SCIProwGetRhs(rows[i]) - constant;
      vals = SCIProwGetVals(rows[i]);
      nnonz = SCIProwGetNNonz(rows[i]);
      cols = SCIProwGetCols(rows[i]);
      
      assert( lhs <= rhs );
      
      /* allocate memory array to be filled with the corresponding subproblem variables */
      SCIP_CALL( SCIPallocBufferArray(scip, &consvars, nnonz) );
      for( j = 0; j < nnonz; j++ ) 
         consvars[j] = subvars[SCIPvarGetProbindex(SCIPcolGetVar(cols[j]))];

      /* create a new linear constraint and add it to the subproblem */
      SCIP_CALL( SCIPcreateConsLinear(subscip, &cons, SCIProwGetName(rows[i]), nnonz, consvars, vals, lhs, rhs,
            TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, TRUE, TRUE, FALSE) );
      SCIP_CALL( SCIPaddCons(subscip, cons) );
      SCIP_CALL( SCIPreleaseCons(subscip, &cons) );
      
      /* free temporary memory */
      SCIPfreeBufferArray(scip, &consvars);
   }

   SCIPfreeBufferArray(scip, &marked);
   return SCIP_OKAY;
}

/** creates a new solution for the original problem by copying the solution of the subproblem */
static
SCIP_RETCODE createNewSol(
   SCIP*                 scip,               /**< original SCIP data structure                        */
   SCIP*                 subscip,            /**< SCIP structure of the subproblem                    */
   SCIP_VAR**            subvars,            /**< the variables of the subproblem                     */
   SCIP_HEUR*            heur,               /**< mutation heuristic structure                        */
   SCIP_SOL*             subsol,             /**< solution of the subproblem                          */
   SCIP_Bool*            success             /**< used to store whether new solution was found or not */
)
{
   SCIP_VAR** vars;                          /* the original problem's variables                */
   int        nvars;
   SCIP_Real* subsolvals;                    /* solution values of the subproblem               */
   SCIP_SOL*  newsol;                        /* solution to be created for the original problem */
        
   assert( scip != NULL );
   assert( subscip != NULL );
   assert( subvars != NULL );
   assert( subsol != NULL );

   /* get variables' data */
   SCIP_CALL( SCIPgetVarsData(scip, &vars, &nvars, NULL, NULL, NULL, NULL) );
   assert( nvars == SCIPgetNOrigVars(subscip) );  
 
   SCIP_CALL( SCIPallocBufferArray(scip, &subsolvals, nvars) );

   /* copy the solution */
   SCIP_CALL( SCIPgetSolVals(subscip, subsol, nvars, subvars, subsolvals) );
       
   /* create new solution for the original problem */
   SCIP_CALL( SCIPcreateSol(scip, &newsol, heur) );
   SCIP_CALL( SCIPsetSolVals(scip, newsol, nvars, vars, subsolvals) );

   /* try to add new solution to scip and free it immediately */
   SCIP_CALL( SCIPtrySolFree(scip, &newsol, TRUE, TRUE, TRUE, success) );

   SCIPfreeBufferArray(scip, &subsolvals);

   return SCIP_OKAY;
}



/*
 * Callback methods of primal heuristic
 */


/** destructor of primal heuristic to free user data (called when SCIP is exiting) */
static
SCIP_DECL_HEURFREE(heurFreeMutation)
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert( heur != NULL );
   assert( scip != NULL );

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert( heurdata != NULL );

   /* free heuristic data */
   SCIPfreeMemory(scip, &heurdata);
   SCIPheurSetData(heur, NULL);

   return SCIP_OKAY;
}

/** initialization method of primal heuristic (called after problem was transformed) */
static
SCIP_DECL_HEURINIT(heurInitMutation)
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert( heur != NULL );
   assert( scip != NULL );

   /* get heuristic's data */
   heurdata = SCIPheurGetData(heur);
   assert( heurdata != NULL );

   /* initialize data */
   heurdata->usednodes = 0;
   heurdata->randseed = 0;

   return SCIP_OKAY;
}


/** deinitialization method of primal heuristic (called before transformed problem is freed) */
#define heurExitMutation NULL

/** solving process initialization method of primal heuristic (called when branch and bound process is about to begin) */
#define heurInitsolMutation NULL

/** solving process deinitialization method of primal heuristic (called before branch and bound process data is freed) */
#define heurExitsolMutation NULL



/** execution method of primal heuristic */
static
SCIP_DECL_HEUREXEC(heurExecMutation)
{  /*lint --e{715}*/
   SCIP_Longint maxnnodes;                  
   SCIP_Longint nsubnodes;                   /* node limit for the subproblem                       */

   SCIP_HEURDATA* heurdata;                  /* heuristic's data                                    */
   SCIP* subscip;                            /* the subproblem created by mutation                  */
   SCIP_VAR** vars;                          /* original problem's variables                        */
   SCIP_VAR** subvars;                       /* subproblem's variables                              */

   SCIP_Real cutoff;                         /* objective cutoff for the subproblem                 */
   SCIP_Real maxnnodesr;
   SCIP_Real memorylimit;
   SCIP_Real timelimit;                      /* timelimit for the subproblem                        */
   SCIP_Real upperbound;

   int nvars;                                /* number of original problem's variables              */
   int i;   

   SCIP_Bool success;     

#ifdef NDEBUG
   SCIP_RETCODE retstat;
#endif

   assert( heur != NULL );
   assert( scip != NULL );
   assert( result != NULL );

   /* get heuristic's data */
   heurdata = SCIPheurGetData(heur);
   assert( heurdata != NULL );
   
   *result = SCIP_DELAYED;

   /* only call heuristic, if feasible solution is available */
   if( SCIPgetNSols(scip) <= 0 )
      return SCIP_OKAY;

   /* only call heuristic, if the best solution comes from transformed problem */
   assert( SCIPgetBestSol(scip) != NULL );
   if( SCIPsolGetOrigin(SCIPgetBestSol(scip)) == SCIP_SOLORIGIN_ORIGINAL )
      return SCIP_OKAY;

   /* only call heuristic, if enough nodes were processed since last incumbent */
   if( SCIPgetNNodes(scip) - SCIPgetSolNodenum(scip,SCIPgetBestSol(scip))  < heurdata->nwaitingnodes)
      return SCIP_OKAY;

   *result = SCIP_DIDNOTRUN;

   /* calculate the maximal number of branching nodes until heuristic is aborted */
   maxnnodesr = heurdata->nodesquot * SCIPgetNNodes(scip);

   /* reward mutation if it succeeded often, count the setup costs for the sub-MIP as 100 nodes */
   maxnnodesr *= 1.0 + 2.0 * (SCIPheurGetNBestSolsFound(heur)+1.0)/(SCIPheurGetNCalls(heur) + 1.0);
   maxnnodes = (SCIP_Longint) maxnnodesr - 100 * SCIPheurGetNCalls(heur);
   maxnnodes += heurdata->nodesofs;

   /* determine the node limit for the current process */
   nsubnodes = maxnnodes - heurdata->usednodes;
   nsubnodes = MIN(nsubnodes, heurdata->maxnodes);

   /* check whether we have enough nodes left to call subproblem solving */
   if( nsubnodes < heurdata->minnodes )
       return SCIP_OKAY;
 
  /* check whether there is enough time and memory left */
   SCIP_CALL( SCIPgetRealParam(scip, "limits/time", &timelimit) );
   if( !SCIPisInfinity(scip, timelimit) )
      timelimit -= SCIPgetSolvingTime(scip);
   SCIP_CALL( SCIPgetRealParam(scip, "limits/memory", &memorylimit) );
   if( !SCIPisInfinity(scip, memorylimit) )   
      memorylimit -= SCIPgetMemUsed(scip)/1048576.0;
   if( timelimit < 10.0 || memorylimit <= 0.0 )
      return SCIP_OKAY;

   if( SCIPisStopped(scip) )
      return SCIP_OKAY;

   *result = SCIP_DIDNOTFIND;

   SCIP_CALL( SCIPgetVarsData(scip, &vars, &nvars, NULL, NULL, NULL, NULL) );

   /* initializing the subproblem */  
   SCIP_CALL( SCIPallocBufferArray(scip, &subvars, nvars) ); 
   SCIP_CALL( SCIPcreate(&subscip) );
   SCIP_CALL( SCIPincludeDefaultPlugins(subscip) );
 
   success = FALSE;
   /* create a new problem, which fixes variables with same value in bestsol and LP relaxation */
   SCIP_CALL( createSubproblem(scip, subscip, subvars, heurdata->minfixingrate, &heurdata->randseed) );
   
   /* do not abort subproblem on CTRL-C */
   SCIP_CALL( SCIPsetBoolParam(subscip, "misc/catchctrlc", FALSE) );
 
   /* disable output to console */
   SCIP_CALL( SCIPsetIntParam(subscip, "display/verblevel", 0) );
  
   /* set limits for the subproblem */
   SCIP_CALL( SCIPsetLongintParam(subscip, "limits/nodes", nsubnodes) ); 
   SCIP_CALL( SCIPsetRealParam(subscip, "limits/time", timelimit) );
   SCIP_CALL( SCIPsetRealParam(subscip, "limits/memory", memorylimit) );

   /* forbid recursive call of heuristics solving subMIPs */
   SCIP_CALL( SCIPsetIntParam(subscip, "heuristics/dins/freq", -1) ); 
   SCIP_CALL( SCIPsetIntParam(subscip, "heuristics/undercover/freq", -1) );
   SCIP_CALL( SCIPsetIntParam(subscip, "heuristics/rins/freq", -1) ); 
   SCIP_CALL( SCIPsetIntParam(subscip, "heuristics/rens/freq", -1) ); 
   SCIP_CALL( SCIPsetIntParam(subscip, "heuristics/localbranching/freq", -1) );
   SCIP_CALL( SCIPsetIntParam(subscip, "heuristics/mutation/freq", -1) );
   SCIP_CALL( SCIPsetIntParam(subscip, "heuristics/crossover/freq", -1) );
   SCIP_CALL( SCIPsetIntParam(subscip, "separating/rapidlearning/freq", -1) );

   /* disable cut separation in sub problem */
   SCIP_CALL( SCIPsetIntParam(subscip, "separating/maxrounds", 0) );
   SCIP_CALL( SCIPsetIntParam(subscip, "separating/maxroundsroot", 0) );
   SCIP_CALL( SCIPsetIntParam(subscip, "separating/maxcuts", 0) ); 
   SCIP_CALL( SCIPsetIntParam(subscip, "separating/maxcutsroot", 0) );
   
   /* use inference branching */
   SCIP_CALL( SCIPsetIntParam(subscip, "branching/inference/priority", INT_MAX/4) );

   /* use best estimate node selection */
   SCIP_CALL( SCIPsetIntParam(subscip, "nodeselection/estimate/stdpriority", INT_MAX/4) ); 

   /* disable expensive presolving */
   SCIP_CALL( SCIPsetIntParam(subscip, "presolving/probing/maxrounds", 0) );
   SCIP_CALL( SCIPsetBoolParam(subscip, "constraints/linear/presolpairwise", FALSE) );
   SCIP_CALL( SCIPsetBoolParam(subscip, "constraints/setppc/presolpairwise", FALSE) );
   SCIP_CALL( SCIPsetBoolParam(subscip, "constraints/logicor/presolpairwise", FALSE) );
   SCIP_CALL( SCIPsetRealParam(subscip, "constraints/linear/maxaggrnormscale", 0.0) );

   /* disable conflict analysis */
   SCIP_CALL( SCIPsetBoolParam(subscip, "conflict/useprop", FALSE) ); 
   SCIP_CALL( SCIPsetBoolParam(subscip, "conflict/useinflp", FALSE) );
   SCIP_CALL( SCIPsetBoolParam(subscip, "conflict/useboundlp", FALSE) );
   SCIP_CALL( SCIPsetBoolParam(subscip, "conflict/usesb", FALSE) ); 
   SCIP_CALL( SCIPsetBoolParam(subscip, "conflict/usepseudo", FALSE) );

   /* add an objective cutoff */
   cutoff = SCIPinfinity(scip);
   assert( !SCIPisInfinity(scip,SCIPgetUpperbound(scip)) );   

   upperbound = SCIPgetUpperbound(scip) - SCIPsumepsilon(scip);
   if( !SCIPisInfinity(scip,-1.0*SCIPgetLowerbound(scip)) )
   {
      cutoff = (1-heurdata->minimprove)*SCIPgetUpperbound(scip) + heurdata->minimprove*SCIPgetLowerbound(scip);
   }
   else
   {
      if ( SCIPgetUpperbound ( scip ) >= 0 )
         cutoff = ( 1 - heurdata->minimprove ) * SCIPgetUpperbound ( scip );
      else
         cutoff = ( 1 + heurdata->minimprove ) * SCIPgetUpperbound ( scip );
   }
   cutoff = MIN(upperbound, cutoff );
   SCIP_CALL( SCIPsetObjlimit(subscip, cutoff) );

   /* solve the subproblem */
   /* Errors in the LP solver should not kill the overall solving process, if the LP is just needed for a heuristic.
    * Hence in optimized mode, the return code is catched and a warning is printed, only in debug mode, SCIP will stop.
    */
#ifdef NDEBUG
   retstat = SCIPsolve(subscip);
   if( retstat != SCIP_OKAY )
   { 
      SCIPwarningMessage("Error while solving subMIP in mutation heuristic; subSCIP terminated with code <%d>\n",
         retstat);
   }
#else
   SCIP_CALL( SCIPsolve(subscip) );
#endif
   
   heurdata->usednodes += SCIPgetNNodes(subscip);

   /* check, whether a solution was found */
   if( SCIPgetNSols(subscip) > 0 )
   {
      SCIP_SOL** subsols;
      int nsubsols;

      /* check, whether a solution was found;
       * due to numerics, it might happen that not all solutions are feasible -> try all solutions until one was accepted
       */
      nsubsols = SCIPgetNSols(subscip);
      subsols = SCIPgetSols(subscip);
      success = FALSE;
      for( i = 0; i < nsubsols && !success; ++i )
      {
         SCIP_CALL( createNewSol(scip, subscip, subvars, heur, subsols[i], &success) );
      }
      if( success )
         *result = SCIP_FOUNDSOL;
   }
   
   /* free subproblem */
   SCIP_CALL( SCIPfreeTransform(subscip) );
   for( i = 0; i < nvars; i++ )
   {
      SCIP_CALL( SCIPreleaseVar(subscip, &subvars[i]) );
   }
   SCIPfreeBufferArray(scip, &subvars);
   SCIP_CALL( SCIPfree(&subscip) );

   return SCIP_OKAY;
}

/*
 * primal heuristic specific interface methods
 */

/** creates the mutation primal heuristic and includes it in SCIP */
SCIP_RETCODE SCIPincludeHeurMutation(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_HEURDATA* heurdata;

   /* create heuristic data */
   SCIP_CALL( SCIPallocMemory(scip, &heurdata) );

   /* include primal heuristic */ 
   SCIP_CALL( SCIPincludeHeur(scip, HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ, HEUR_FREQOFS,
         HEUR_MAXDEPTH, HEUR_TIMING,
         heurFreeMutation, heurInitMutation, heurExitMutation,
         heurInitsolMutation, heurExitsolMutation, heurExecMutation,
         heurdata) );
  
   /* add mutation primal heuristic parameters */ 
   SCIP_CALL( SCIPaddIntParam(scip, "heuristics/mutation/nodesofs",
         "number of nodes added to the contingent of the total nodes",
         &heurdata->nodesofs, FALSE, DEFAULT_NODESOFS, 0, INT_MAX, NULL, NULL) );
   
   SCIP_CALL( SCIPaddIntParam(scip, "heuristics/mutation/maxnodes",
         "maximum number of nodes to regard in the subproblem",
         &heurdata->maxnodes, TRUE, DEFAULT_MAXNODES, 0, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddIntParam(scip, "heuristics/mutation/minnodes",
         "minimum number of nodes required to start the subproblem",
         &heurdata->minnodes, TRUE, DEFAULT_MINNODES, 0, INT_MAX, NULL, NULL) );
   
   SCIP_CALL( SCIPaddIntParam(scip, "heuristics/mutation/nwaitingnodes",
         "number of nodes without incumbent change that heuristic should wait",
         &heurdata->nwaitingnodes, TRUE, DEFAULT_NWAITINGNODES, 0, INT_MAX, NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(scip, "heuristics/mutation/nodesquot",
         "contingent of sub problem nodes in relation to the number of nodes of the original problem",
         &heurdata->nodesquot, FALSE, DEFAULT_NODESQUOT, 0.0, 1.0, NULL, NULL) );
   
   SCIP_CALL( SCIPaddRealParam(scip, "heuristics/mutation/minfixingrate",
         "percentage of integer variables that have to be fixed ",
         &heurdata->minfixingrate, FALSE, DEFAULT_MINFIXINGRATE, SCIPsumepsilon(scip), 1.0-SCIPsumepsilon(scip), NULL, NULL) );

   SCIP_CALL( SCIPaddRealParam(scip, "heuristics/mutation/minimprove",
         "factor by which Mutation should at least improve the incumbent",
         &heurdata->minimprove, TRUE, DEFAULT_MINIMPROVE, 0.0, 1.0, NULL, NULL) );
   
   return SCIP_OKAY;
}
