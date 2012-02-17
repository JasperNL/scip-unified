/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2012 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#define SCIP_DEBUG
#define STATISTIC_INFORMATION

/* uncomment to get statistical output at the end of SCIP run */
/* #define STATISTIC_INFORMATION */

/**@file   heur_nlpfracdiving.c
 * @brief  NLP diving heuristic that chooses fixings w.r.t. the fractionalities
 * @author Timo Berthold
 * @author Stefan Vigerske
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "scip/heur_nlpfracdiving.h"
#include "scip/heur_subnlp.h" /* for NLP initialization */
#include "scip/heur_undercover.h" /* for cover computation */
#include "nlpi/nlpi.h" /* for NLP statistics, currently */


#define HEUR_NAME             "nlpfracdiving"
#define HEUR_DESC             "NLP diving heuristic that chooses fixings w.r.t. the fractionalities"
#define HEUR_DISPCHAR         '%'
#define HEUR_PRIORITY         -1003000
#define HEUR_FREQ             10
#define HEUR_FREQOFS          3
#define HEUR_MAXDEPTH         -1
#define HEUR_TIMING           SCIP_HEURTIMING_AFTERLPPLUNGE
#define HEUR_USESSUBSCIP      FALSE  /**< does the heuristic use a secondary SCIP instance? */

/* event handler properties */
#define EVENTHDLR_NAME         "Nlpfracdiving"
#define EVENTHDLR_DESC         "bound change event handler for "HEUR_NAME" heuristic"


/*
 * Default parameter settings
 */

#define DEFAULT_MINRELDEPTH         0.0 /**< minimal relative depth to start diving */
#define DEFAULT_MAXRELDEPTH         1.0 /**< maximal relative depth to start diving */
#define DEFAULT_MAXNLPITERQUOT     0.05 /**< maximal fraction of diving LP iterations compared to node NLP iterations */
#define DEFAULT_MAXNLPITEROFS      1000 /**< additional number of allowed NLP iterations */
#define DEFAULT_MAXDIVEUBQUOT       0.8 /**< maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound)
                                         *   where diving is performed (0.0: no limit) */
#define DEFAULT_MAXDIVEAVGQUOT      0.0 /**< maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound)
                                         *   where diving is performed (0.0: no limit) */
#define DEFAULT_MAXDIVEUBQUOTNOSOL  0.1 /**< maximal UBQUOT when no solution was found yet (0.0: no limit) */
#define DEFAULT_MAXDIVEAVGQUOTNOSOL 0.0 /**< maximal AVGQUOT when no solution was found yet (0.0: no limit) */
#define DEFAULT_MINSUCCQUOT         0.1 /**< heuristic will not run if less then this percentage of calls succeeded (0.0: no limit) */
#define DEFAULT_FIXQUOT             0.2 /**< percentage of fractional variables that should be fixed before the next NLP solve */
#define DEFAULT_BACKTRACK          TRUE /**< use one level of backtracking if infeasibility is encountered? */
#define DEFAULT_PREFERLPFRACS      TRUE /**< prefer variables that are also fractional in LP solution? */
#define DEFAULT_PREFERCOVER        TRUE /**< should variables in a minimal cover be preferred? */
#define DEFAULT_SOLVESUBMIP       FALSE /**< should a sub-MIP be solved if all cover variables are fixed? */

#define MINNLPITER                 1000 /**< minimal number of NLP iterations allowed in each NLP solving call */

/* enable statistic output by defining macro STATISTIC_INFORMATION */
#ifdef STATISTIC_INFORMATION
#define STATISTIC(x)                {x}
#else
#define STATISTIC(x)             /**/
#endif


/* locally defined heuristic data */
struct SCIP_HeurData
{
   SCIP_SOL*             sol;                /**< working solution */
   SCIP_Real             minreldepth;        /**< minimal relative depth to start diving */
   SCIP_Real             maxreldepth;        /**< maximal relative depth to start diving */
   SCIP_Real             maxnlpiterquot;     /**< maximal fraction of diving NLP iterations compared to node NLP iterations */
   int                   maxnlpiterofs;      /**< additional number of allowed NLP iterations */
   SCIP_Real             maxdiveubquot;      /**< maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound)
                                              *   where diving is performed (0.0: no limit) */
   SCIP_Real             maxdiveavgquot;     /**< maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound)
                                              *   where diving is performed (0.0: no limit) */
   SCIP_Real             maxdiveubquotnosol; /**< maximal UBQUOT when no solution was found yet (0.0: no limit) */
   SCIP_Real             maxdiveavgquotnosol;/**< maximal AVGQUOT when no solution was found yet (0.0: no limit) */
   SCIP_Real             minsuccquot;        /**< heuristic will not run if less then this percentage of calls succeeded (0.0: no limit) */
   SCIP_Real             fixquot;            /**< percentage of fractional variables that should be fixed before the next NLP solve */
   SCIP_Bool             backtrack;          /**< use one level of backtracking if infeasibility is encountered? */
   SCIP_Bool             preferlpfracs;      /**< prefer variables that are also fractional in LP solution? */
   SCIP_Bool             prefercover;        /**< should variables in a minimal cover be preferred? */
   SCIP_Bool             solvesubmip;        /**< should a sub-MIP be solved if all cover variables are fixed? */
   SCIP_Longint          nnlpiterations;     /**< NLP iterations used in this heuristic */
   int                   nsuccess;           /**< number of runs that produced at least one feasible solution */
   int                   nfixedcovervars;    /**< number of variables in the cover that are already fixed */
#ifdef STATISTIC_INFORMATION
   int                   nnlpsolves;         /**< number of NLP solves */
   int                   nfailcutoff;        /**< number of fails due to cutoff */
   int                   nfaildepth;         /**< number of fails due to too deep */
   int                   nfailnlperror;      /**< number of fails due to NLP error */
#endif
   SCIP_EVENTHDLR*       eventhdlr;          /**< event handler for bound change events */
};




/*
 * local methods
 */

/** creates a new solution for the original problem by copying the solution of the subproblem */
static
SCIP_RETCODE createNewSol(
   SCIP*                 scip,               /**< original SCIP data structure                        */
   SCIP*                 subscip,            /**< SCIP structure of the subproblem                    */
   SCIP_HEUR*            heur,               /**< heuristic structure                                 */
   SCIP_HASHMAP*         varmap,             /**< hash map for variables */
   SCIP_SOL*             subsol,             /**< solution of the subproblem                          */
   SCIP_Bool*            success             /**< used to store whether new solution was found or not */
   )
{
   SCIP_VAR** vars;                          /* the original problem's variables                */
   SCIP_VAR** subvars;
   SCIP_Real* subsolvals;                    /* solution values of the subproblem               */
   SCIP_SOL*  newsol;                        /* solution to be created for the original problem */
   int        nvars;                         /* the original problem's number of variables      */
   int i;

   assert(scip != NULL);
   assert(subscip != NULL);
   assert(subsol != NULL);

   /* get variables' data */
   SCIP_CALL( SCIPgetVarsData(scip, &vars, &nvars, NULL, NULL, NULL, NULL) );

   /* sub-SCIP may have more variables than the number of active (transformed) variables in the main SCIP
    * since constraint copying may have required the copy of variables that are fixed in the main SCIP
    */
   assert(nvars <= SCIPgetNOrigVars(subscip));

   SCIP_CALL( SCIPallocBufferArray(scip, &subsolvals, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &subvars, nvars) );

   for( i = 0; i < nvars; i++ )
     subvars[i] = (SCIP_VAR*) SCIPhashmapGetImage(varmap, vars[i]);

   /* copy the solution */
   SCIP_CALL( SCIPgetSolVals(subscip, subsol, nvars, subvars, subsolvals) );

   /* create new solution for the original problem */
   SCIP_CALL( SCIPcreateSol(scip, &newsol, heur) );
   SCIP_CALL( SCIPsetSolVals(scip, newsol, nvars, vars, subsolvals) );

   /* try to add new solution to scip and free it immediately */
   SCIP_CALL( SCIPtrySolFree(scip, &newsol, FALSE, TRUE, TRUE, TRUE, success) );

   SCIPfreeBufferArray(scip, &subvars);
   SCIPfreeBufferArray(scip, &subsolvals);

   return SCIP_OKAY;
}


/** solves subproblem and passes best feasible solution to original SCIP instance */
static
SCIP_RETCODE solveSubMIP(
   SCIP*                 scip,               /**< SCIP data structure of the original problem */
   SCIP_HEUR*            heur,               /**< heuristic data structure */
   SCIP_VAR**            covervars,          /**< variables in the cover, should be fixed locally */
   int                   ncovervars,         /**< number of variables in the cover */
   SCIP_Bool*            success             /**< pointer to store whether a solution was found */
   )
{
   SCIP* subscip;
   SCIP_HASHMAP* varmap;
   SCIP_SOL** subsols;
   SCIP_Real timelimit;
   SCIP_Real memorylimit;
   SCIP_RETCODE retcode;
   int c;
   int nsubsols;
   SCIP_Bool valid;
            
   /* create subproblem */
   SCIP_CALL( SCIPcreate(&subscip) );
            
   /* create the variable mapping hash map */
   SCIP_CALL( SCIPhashmapCreate(&varmap, SCIPblkmem(subscip), SCIPcalcHashtableSize(5 * SCIPgetNVars(scip))) );

   *success = FALSE;
            
   /* copy original problem to subproblem; do not copy pricers */
   SCIP_CALL( SCIPcopy(scip, subscip, varmap, NULL, "undercoversub", FALSE, FALSE, &valid) );

   /* assert that cover variables are fixed in source and target SCIP */
   for( c = 0; c < ncovervars; c++)
   {
      assert(SCIPisFeasEQ(scip, SCIPvarGetLbLocal(covervars[c]), SCIPvarGetUbLocal(covervars[c])));
      assert(SCIPisFeasEQ(scip, SCIPvarGetLbGlobal((SCIP_VAR*) SCIPhashmapGetImage(varmap, covervars[c])),
            SCIPvarGetUbGlobal((SCIP_VAR*) SCIPhashmapGetImage(varmap, covervars[c]))));
   }

   /* set parameters for sub-SCIP */

   /* do not abort subproblem on CTRL-C */
   SCIP_CALL( SCIPsetBoolParam(subscip, "misc/catchctrlc", FALSE) );

   /* disable output to console */
   SCIP_CALL( SCIPsetIntParam(subscip, "display/verblevel", 0) );

   /* check whether there is enough time and memory left */
   timelimit = 0.0;
   memorylimit = 0.0;
   SCIP_CALL( SCIPgetRealParam(scip, "limits/time", &timelimit) );
   if( !SCIPisInfinity(scip, timelimit))
      timelimit -= SCIPgetSolvingTime(scip);
   SCIP_CALL( SCIPgetRealParam(scip, "limits/memory", &memorylimit) );
   if( !SCIPisInfinity(scip, memorylimit) )
      memorylimit -= SCIPgetMemUsed(scip)/1048576.0;
   if( timelimit <= 0.0 || memorylimit <= 0.0 )
      goto TERMINATE;

   /* set limits for the subproblem */
   SCIP_CALL( SCIPsetLongintParam(subscip, "limits/stallnodes", 100) );
   SCIP_CALL( SCIPsetLongintParam(subscip, "limits/nodes", 500) );
   SCIP_CALL( SCIPsetRealParam(subscip, "limits/time", timelimit) );
   SCIP_CALL( SCIPsetRealParam(subscip, "limits/memory", memorylimit) );

   /* forbid recursive call of heuristics and separators solving sub-SCIPs */
   SCIP_CALL( SCIPsetSubscipsOff(subscip, TRUE) );

   /* disable cutting plane separation */
   SCIP_CALL( SCIPsetSeparating(subscip, SCIP_PARAMSETTING_OFF, TRUE) );

   /* disable expensive presolving */
   SCIP_CALL( SCIPsetPresolving(subscip, SCIP_PARAMSETTING_FAST, TRUE) );

   /* use best estimate node selection */
   if( SCIPfindNodesel(scip, "estimate") != NULL )
   {
      SCIP_CALL( SCIPsetIntParam(subscip, "nodeselection/estimate/stdpriority", INT_MAX/4) );
   }

   /* use inference branching */
   if( SCIPfindBranchrule(subscip, "inference") != NULL )
   {
      SCIP_CALL( SCIPsetIntParam(subscip, "branching/inference/priority", INT_MAX/4) );
   }

   /* disable conflict analysis */
   SCIP_CALL( SCIPsetBoolParam(subscip, "conflict/useprop", FALSE) );
   SCIP_CALL( SCIPsetBoolParam(subscip, "conflict/useinflp", FALSE) );
   SCIP_CALL( SCIPsetBoolParam(subscip, "conflict/useboundlp", FALSE) );
   SCIP_CALL( SCIPsetBoolParam(subscip, "conflict/usesb", FALSE) );
   SCIP_CALL( SCIPsetBoolParam(subscip, "conflict/usepseudo", FALSE) );

   if( SCIPgetNSols(scip) > 0 )
   {
      SCIP_Real upperbound;
      SCIP_Real cutoffbound;
      SCIP_Real minimprove;
      
      cutoffbound = SCIPinfinity(scip);
      assert( !SCIPisInfinity(scip,SCIPgetUpperbound(scip)) );

      upperbound = SCIPgetUpperbound(scip) - SCIPsumepsilon(scip);
      minimprove = 0.01;

      if( !SCIPisInfinity(scip,-1.0*SCIPgetLowerbound(scip)) )
      {
         cutoffbound = (1-minimprove)*SCIPgetUpperbound(scip) + minimprove*SCIPgetLowerbound(scip);
      }
      else
      {
         if( SCIPgetUpperbound(scip) >= 0 )
            cutoffbound = (1 - minimprove)*SCIPgetUpperbound(scip);
         else
            cutoffbound = (1 + minimprove)*SCIPgetUpperbound(scip);
      }
      cutoffbound = MIN(upperbound, cutoffbound);
      SCIP_CALL( SCIPsetObjlimit(subscip, cutoffbound) );
   }

#ifdef SCIP_DEBUG
   /* for debugging, enable sub-SCIP output */
   SCIP_CALL( SCIPsetIntParam(subscip, "display/verblevel", 5) );
   SCIP_CALL( SCIPsetIntParam(subscip, "display/freq", 100000000) );
#endif

   retcode = SCIPsolve(subscip);

   /* Errors in solving the subproblem should not kill the overall solving process
    * Hence, the return code is caught and a warning is printed, only in debug mode, SCIP will stop.
    */
   if( retcode != SCIP_OKAY )
   {
#ifndef NDEBUG
      SCIP_CALL( retcode );
#endif
      SCIPwarningMessage("Error while solving subproblem in "HEUR_NAME" heuristic; sub-SCIP terminated with code <%d>\n",retcode);
   }

   /* check, whether a solution was found;
    * due to numerics, it might happen that not all solutions are feasible -> try all solutions until one was accepted
    */
   nsubsols = SCIPgetNSols(subscip);
   subsols = SCIPgetSols(subscip);
   for( c = 0; c < nsubsols && !(*success); ++c )
   {
      SCIP_CALL( createNewSol(scip, subscip, heur, varmap, subsols[c], success) );
   }

 TERMINATE:
   /* free sub-SCIP and hash map */
   SCIP_CALL( SCIPfree(&subscip) );
   SCIPhashmapFree(&varmap);

   return SCIP_OKAY;
}

/* ---------------- Callback methods of event handler ---------------- */

/* exec the event handler
 *
 * We update the number of variables fixed in the cover
 */
static
SCIP_DECL_EVENTEXEC(eventExecNlpFracdiving)
{
   SCIP_EVENTTYPE eventtype;
   SCIP_HEURDATA* heurdata;
   SCIP_VAR* var;

   SCIP_Real oldbound;
   SCIP_Real newbound;
   SCIP_Real otherbound;
   
   assert(eventhdlr != NULL);
   assert(eventdata != NULL);
   assert(strcmp(SCIPeventhdlrGetName(eventhdlr), EVENTHDLR_NAME) == 0);
   assert(event != NULL);

   heurdata = (SCIP_HEURDATA*)eventdata;
   assert(heurdata != NULL);
   assert(0 <= heurdata->nfixedcovervars && heurdata->nfixedcovervars <= SCIPgetNVars(scip));

   oldbound = SCIPeventGetOldbound(event);
   newbound = SCIPeventGetNewbound(event);
   var = SCIPeventGetVar(event);

   eventtype = SCIPeventGetType(event);
   otherbound = (eventtype & SCIP_EVENTTYPE_LBCHANGED) ? SCIPvarGetUbLocal(var) : SCIPvarGetLbLocal(var);

   switch( eventtype )
   {
   case SCIP_EVENTTYPE_LBTIGHTENED:
   case SCIP_EVENTTYPE_UBTIGHTENED:
      /* if cover variable is now fixed */
         if( SCIPisFeasEQ(scip, newbound, otherbound) )
         {
            assert(!SCIPisFeasEQ(scip, oldbound, otherbound));
            ++(heurdata->nfixedcovervars);
         }
         break;
   case SCIP_EVENTTYPE_LBRELAXED:
   case SCIP_EVENTTYPE_UBRELAXED:
      /* if cover variable is now unfixed */
      if( SCIPisFeasEQ(scip, oldbound,otherbound) )
      {
         assert(!SCIPisFeasEQ(scip, newbound, otherbound));
         --(heurdata->nfixedcovervars);
      }
      break;
   default:
      SCIPerrorMessage("invalid event type.\n");
      return SCIP_INVALIDDATA;
   }
   assert(0 <= heurdata->nfixedcovervars && heurdata->nfixedcovervars <= SCIPgetNVars(scip));

   SCIPdebugMessage("changed bound of cover variable <%s> from %f to %f (nfixedcovervars: %d).\n", SCIPvarGetName(var),
                    oldbound, newbound, heurdata->nfixedcovervars);

   return SCIP_OKAY;
}


/*
 * Callback methods
 */

/** copy method for primal heuristic plugins (called when SCIP copies plugins) */
static
SCIP_DECL_HEURCOPY(heurCopyNlpFracdiving)
{  /*lint --e{715}*/
   assert(scip != NULL);
   assert(heur != NULL);
   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);

   /* call inclusion method of primal heuristic */
   /* @todo disabled copying for easier development/debugging */
/*   SCIP_CALL( SCIPincludeHeurNlpFracdiving(scip) ); */

   return SCIP_OKAY;
}

/** destructor of primal heuristic to free user data (called when SCIP is exiting) */
static
SCIP_DECL_HEURFREE(heurFreeNlpFracdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);
   assert(scip != NULL);

   /* free heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);
   SCIPfreeMemory(scip, &heurdata);
   SCIPheurSetData(heur, NULL);

   return SCIP_OKAY;
}


/** initialization method of primal heuristic (called after problem was transformed) */
static
SCIP_DECL_HEURINIT(heurInitNlpFracdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* create working solution */
   SCIP_CALL( SCIPcreateSol(scip, &heurdata->sol, heur) );

   /* initialize data */
   heurdata->nnlpiterations = 0;
   heurdata->nsuccess = 0;
   heurdata->nfixedcovervars = 0;
   STATISTIC(
      heurdata->nnlpsolves = 0;
      heurdata->nfailcutoff = 0;
      heurdata->nfaildepth = 0;
      heurdata->nfailnlperror = 0;
      );

   return SCIP_OKAY;
}


/** deinitialization method of primal heuristic (called before transformed problem is freed) */
static
SCIP_DECL_HEUREXIT(heurExitNlpFracdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;

   assert(heur != NULL);
   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);

   /* get heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* free working solution */
   SCIP_CALL( SCIPfreeSol(scip, &heurdata->sol) );

   STATISTIC(
      if( strstr(SCIPgetProbName(scip), "_covering") == NULL && SCIPheurGetNCalls(heur) > 0 )
      {
         SCIPinfoMessage(scip, NULL, "%-20s %5"SCIP_LONGINT_FORMAT" sols in %5"SCIP_LONGINT_FORMAT" runs, %6.1fs, %7"SCIP_LONGINT_FORMAT" NLP iters in %5d NLP solves, %5.1f avg., %3d%% success %3d%% cutoff %3d%% depth %3d%% nlperror\n",
            SCIPgetProbName(scip), SCIPheurGetNSolsFound(heur), SCIPheurGetNCalls(heur), SCIPheurGetTime(heur),
            heurdata->nnlpiterations, heurdata->nnlpsolves, heurdata->nnlpiterations/MAX(1.0,(SCIP_Real)heurdata->nnlpsolves),
            (100*heurdata->nsuccess) / (int)SCIPheurGetNCalls(heur), (100*heurdata->nfailcutoff) / (int)SCIPheurGetNCalls(heur), (100*heurdata->nfaildepth) / (int)SCIPheurGetNCalls(heur), (100*heurdata->nfailnlperror) / (int)SCIPheurGetNCalls(heur)
         );
      }
      );

   return SCIP_OKAY;
}


/** solving process initialization method of primal heuristic (called when branch and bound process is about to begin) */
static
SCIP_DECL_HEURINITSOL(heurInitsolNlpFracdiving)
{  /*lint --e{715}*/
   SCIP_HEUR* nlpheur;

   if( !SCIPisNLPConstructed(scip) )
      return SCIP_OKAY;

   /* find NLP local search heuristic */
   nlpheur = SCIPfindHeur(scip, "subnlp");

   /* add global linear constraints to NLP relaxation */
   if( nlpheur != NULL )
   {
      SCIP_CALL( SCIPaddLinearConsToNlpHeurSubNlp(scip, nlpheur, TRUE, TRUE) );
   }

   return SCIP_OKAY;
}

/** solving process deinitialization method of primal heuristic (called before branch and bound process data is freed) */
#define heurExitsolNlpFracdiving NULL


/** execution method of primal heuristic */
static
SCIP_DECL_HEUREXEC(heurExecNlpFracdiving) /*lint --e{715}*/
{  /*lint --e{715}*/
   SCIP_HEURDATA* heurdata;
   SCIP_NLPSOLSTAT nlpsolstat;
   SCIP_LPSOLSTAT lpsolstat;
   SCIP_VAR* var;
   SCIP_VAR** nlpcands;
   SCIP_VAR** covervars;
   SCIP_Real* nlpcandssol;
   SCIP_Real* nlpcandsfrac;
   SCIP_HASHMAP* varincover;
   SCIP_Real searchubbound;
   SCIP_Real searchavgbound;
   SCIP_Real searchbound;
   SCIP_Real objval;
   SCIP_Real oldobjval;
   SCIP_Real obj;
   SCIP_Real objgain;
   SCIP_Real bestobjgain;
   SCIP_Real frac;
   SCIP_Real bestfrac;
   SCIP_Bool bestcandmayrounddown;
   SCIP_Bool bestcandmayroundup;
   SCIP_Bool bestcandroundup;
   SCIP_Bool mayrounddown;
   SCIP_Bool mayroundup;
   SCIP_Bool roundup;
   SCIP_Bool nlperror;
   SCIP_Bool lperror;
   SCIP_Bool cutoff;
   SCIP_Bool backtracked;
   SCIP_Bool solvenlp;
   SCIP_Bool covercomputed;
   SCIP_Bool solvesubmip;
   SCIP_Longint ncalls;
   SCIP_Longint nsolsfound;
   SCIP_Longint nnlpiterations;
   SCIP_Longint maxnnlpiterations;
   int npseudocands;
   int nlpbranchcands;
   int ncovervars;
   int nnlpcands;
   int startnnlpcands;
   int depth;
   int maxdepth;
   int maxdivedepth;
   int divedepth;
   int lastnlpsolvedepth;
   int bestcand;
   int c;
   int       backtrackdepth;   /* depth where to go when backtracking */
   SCIP_VAR* backtrackvar;     /* (first) variable to fix differently in backtracking */
   SCIP_Real backtrackvarval;  /* (fractional) value of backtrack variable */
   SCIP_Bool backtrackroundup; /* whether variable should be rounded up in backtracking */

   backtrackdepth = -1;
   backtrackvar = NULL;
   backtrackvarval = 0.0;
   backtrackroundup = FALSE;

   assert(heur != NULL);
   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);
   assert(scip != NULL);
   assert(result != NULL);
   /* assert(SCIPhasCurrentNodeLP(scip)); */

   *result = SCIP_DIDNOTRUN;

   /* only call heuristic, if an NLP relaxation has been constructed */
   if( !SCIPisNLPConstructed(scip) || SCIPgetNNlpis(scip) == 0 )
      return SCIP_OKAY;

   /* get heuristic's data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   /* do not call heuristic, if it barely succeded */
   if( (SCIPheurGetNSolsFound(heur)+1) / (SCIP_Real)(SCIPheurGetNCalls(heur)+1) < heurdata->minsuccquot )
      return SCIP_OKAY;

   *result = SCIP_DELAYED;
#if 0
   /* only call heuristic, if an optimal LP solution is at hand */
   if( SCIPgetLPSolstat(scip) != SCIP_LPSOLSTAT_OPTIMAL )
      return SCIP_OKAY;

   /* only call heuristic, if the LP solution is basic (which allows fast resolve in diving) */
   if( !SCIPisLPSolBasic(scip) )
      return SCIP_OKAY;
#endif
   /* don't dive two times at the same node */
   if( SCIPgetLastDivenode(scip) == SCIPgetNNodes(scip) && SCIPgetDepth(scip) > 0 )
      return SCIP_OKAY;

   *result = SCIP_DIDNOTRUN;

   /* only try to dive, if we are in the correct part of the tree, given by minreldepth and maxreldepth */
   depth = SCIPgetDepth(scip);
   maxdepth = SCIPgetMaxDepth(scip);
   maxdepth = MAX(maxdepth, 30);
   if( depth < heurdata->minreldepth*maxdepth || depth > heurdata->maxreldepth*maxdepth )
      return SCIP_OKAY;

   /* calculate the maximal number of NLP iterations until heuristic is aborted */
   nnlpiterations = 100; /*TODO was SCIPgetNNodeLPIterations(scip); */
   ncalls = SCIPheurGetNCalls(heur);
   nsolsfound = 10*SCIPheurGetNBestSolsFound(heur) + heurdata->nsuccess;
   maxnnlpiterations = (SCIP_Longint)((1.0 + 10.0*(nsolsfound+1.0)/(ncalls+1.0)) * heurdata->maxnlpiterquot * nnlpiterations);
   maxnnlpiterations += heurdata->maxnlpiterofs;

   /* don't try to dive, if we took too many NLP iterations during diving */
   if( heurdata->nnlpiterations >= maxnnlpiterations )
      return SCIP_OKAY;

   /* allow at least a certain number of NLP iterations in this dive */
   maxnnlpiterations = MAX(maxnnlpiterations, heurdata->nnlpiterations + MINNLPITER);

   /* don't try to dive, if there are no unfixed discrete variables */
   SCIP_CALL( SCIPgetPseudoBranchCands(scip, NULL, &npseudocands, NULL) );
   if( npseudocands == 0 )
      return SCIP_OKAY;

   *result = SCIP_DIDNOTFIND;

#if 0 /* def SCIP_DEBUG */
   SCIP_CALL( SCIPsetNLPIntPar(scip, SCIP_NLPPAR_VERBLEVEL, 1) );
#endif
   /* @todo reset feastol when heuristic finished */
   SCIP_CALL( SCIPsetNLPRealPar(scip, SCIP_NLPPAR_FEASTOL, 0.01*SCIPfeastol(scip)) );

   /* set iteration limit; @todo reset limit when heuristic finished */
   SCIP_CALL( SCIPsetNLPIntPar(scip, SCIP_NLPPAR_ITLIM, maxnnlpiterations) );

   /* set starting point to lp solution */
   SCIP_CALL( SCIPsetNLPInitialGuessSol(scip, NULL) );

   /* solve NLP relaxation */
   SCIP_CALL( SCIPsolveNLP(scip) );
   STATISTIC( ++heurdata->nnlpsolves; )

   /* give up, if no feasible solution found */
   nlpsolstat = SCIPgetNLPSolstat(scip);
   if( nlpsolstat >= SCIP_NLPSOLSTAT_LOCINFEASIBLE )
   {
      SCIPdebugMessage("initial NLP infeasible or not solvable --> stop\n");

      /* update iteration count */
      if( SCIPgetNLPTermstat(scip) < SCIP_NLPTERMSTAT_NUMERR )
      {
         SCIP_NLPSTATISTICS* nlpstatistics;

         SCIP_CALL( SCIPnlpStatisticsCreate(&nlpstatistics) );
         SCIP_CALL( SCIPgetNLPStatistics(scip, nlpstatistics) );
         heurdata->nnlpiterations += SCIPnlpStatisticsGetNIterations(nlpstatistics);
         SCIPnlpStatisticsFree(&nlpstatistics);

         STATISTIC( heurdata->nfailcutoff++; )
      }
      else
      {
         STATISTIC( heurdata->nfailnlperror++; )
      }

      return SCIP_OKAY;
   }

   /* get fractional variables that should be integral */
   SCIP_CALL( SCIPgetNLPFracVars(scip, &nlpcands, &nlpcandssol, &nlpcandsfrac, &nnlpcands, NULL) );

   lpsolstat = SCIPgetLPSolstat(scip);
   if( lpsolstat == SCIP_LPSOLSTAT_OPTIMAL )
      nlpbranchcands = SCIPgetNLPBranchCands(scip);
   else
      nlpbranchcands = 0;

   /* prefer decisions on variables which are also fractional in LP solution */
   if( heurdata->preferlpfracs && lpsolstat == SCIP_LPSOLSTAT_OPTIMAL )
      for( c = 0; c < nnlpcands; ++c )
      {
         var = nlpcands[c];
         if( SCIPisFeasIntegral(scip, SCIPgetSolVal(scip, NULL, var)) )
            nlpcandsfrac[c] *= 100.0;
      }

   /* don't try to dive, if there are no fractional variables */
   if( nnlpcands == 0 )
      return SCIP_OKAY;

   /* calculate the objective search bound */
   if( SCIPgetNSolsFound(scip) == 0 )
   {
      if( heurdata->maxdiveubquotnosol > 0.0 )
         searchubbound = SCIPgetLowerbound(scip)
            + heurdata->maxdiveubquotnosol * (SCIPgetCutoffbound(scip) - SCIPgetLowerbound(scip));
      else
         searchubbound = SCIPinfinity(scip);
      if( heurdata->maxdiveavgquotnosol > 0.0 )
         searchavgbound = SCIPgetLowerbound(scip)
            + heurdata->maxdiveavgquotnosol * (SCIPgetAvgLowerbound(scip) - SCIPgetLowerbound(scip));
      else
         searchavgbound = SCIPinfinity(scip);
   }
   else
   {
      if( heurdata->maxdiveubquot > 0.0 )
         searchubbound = SCIPgetLowerbound(scip)
            + heurdata->maxdiveubquot * (SCIPgetCutoffbound(scip) - SCIPgetLowerbound(scip));
      else
         searchubbound = SCIPinfinity(scip);
      if( heurdata->maxdiveavgquot > 0.0 )
         searchavgbound = SCIPgetLowerbound(scip)
            + heurdata->maxdiveavgquot * (SCIPgetAvgLowerbound(scip) - SCIPgetLowerbound(scip));
      else
         searchavgbound = SCIPinfinity(scip);
   }
   searchbound = MIN(searchubbound, searchavgbound);
   if( SCIPisObjIntegral(scip) )
      searchbound = SCIPceil(scip, searchbound);

   /* calculate the maximal diving depth: 10 * min{number of integer variables, max depth} */
   maxdivedepth = SCIPgetNBinVars(scip) + SCIPgetNIntVars(scip);
   maxdivedepth = MIN(maxdivedepth, maxdepth);
   maxdivedepth *= 10;

   covercomputed = FALSE;
   varincover = NULL;
   
   /* compute cover, if required */
   if( heurdata->prefercover || heurdata->solvesubmip )
   {
      SCIP_Real timelimit;
      SCIP_Real memorylimit;

      /* get limits */
      SCIP_CALL( SCIPgetRealParam(scip, "limits/time", &timelimit) );
      SCIP_CALL( SCIPgetRealParam(scip, "limits/memory", &memorylimit) );
      if( !SCIPisInfinity(scip, timelimit) )
         timelimit -= SCIPgetSolvingTime(scip);
      if( !SCIPisInfinity(scip, memorylimit) )
         memorylimit -= SCIPgetMemUsed(scip)/1048576.0;

      /* compute cover */
      SCIP_CALL( SCIPallocBufferArray(scip, &covervars, SCIPgetNVars(scip)) );
      SCIP_CALL( SCIPcomputeCoverUndercover(scip, &ncovervars, covervars, timelimit, memorylimit, FALSE, FALSE, FALSE, 'u', &covercomputed) );

      if( covercomputed && heurdata->solvesubmip )
      {
         /* create hash map */
         SCIP_CALL( SCIPhashmapCreate(&varincover, SCIPblkmem(scip), SCIPcalcHashtableSize(2 * ncovervars)) );

         /* process variables in the cover */
         for( c = 0; c < ncovervars; c++ )
         {
            /* insert variable into hash map */
            if( SCIPvarGetType(covervars[c]) < SCIP_VARTYPE_IMPLINT )
            {
               assert(!SCIPhashmapExists(varincover, covervars[c]));
               SCIP_CALL( SCIPhashmapInsert(varincover, covervars[c], (void*) (size_t) (c+1)) );
            }

            /* catch bound change events of cover variables */
            assert(heurdata->eventhdlr != NULL);
            SCIP_CALL( SCIPcatchVarEvent(scip, covervars[c], SCIP_EVENTTYPE_BOUNDCHANGED, heurdata->eventhdlr,
                  (SCIP_EVENTDATA*) heurdata, NULL) );
            assert(!SCIPisFeasEQ(scip, SCIPvarGetLbLocal(covervars[c]), SCIPvarGetUbLocal(covervars[c])));
         }
      }
   }

   /* start diving */
   SCIP_CALL( SCIPstartProbing(scip) );

   /* get NLP objective value*/
   objval = SCIPgetNLPObjval(scip);

   SCIPdebugMessage("(node %"SCIP_LONGINT_FORMAT") executing nlpfracdiving heuristic: depth=%d, %d fractionals, dualbound=%g, searchbound=%g\n",
      SCIPgetNNodes(scip), SCIPgetDepth(scip), nnlpcands, SCIPgetDualbound(scip), SCIPretransformObj(scip, searchbound));

   /* dive as long we are in the given objective, depth and iteration limits and fractional variables exist, but
    * - if possible, we dive at least with the depth 10
    * - if the number of fractional variables decreased at least with 1 variable per 2 dive depths, we continue diving
    */
   nlperror = FALSE;
   lperror = FALSE;
   cutoff = FALSE;
   divedepth = 0;
   lastnlpsolvedepth = 0;
   bestcandmayrounddown = FALSE;
   bestcandmayroundup = FALSE;
   startnnlpcands = nnlpcands;
   solvesubmip = heurdata->solvesubmip;
   
   while( !nlperror && !cutoff && (nlpsolstat <= SCIP_NLPSOLSTAT_FEASIBLE || nlpsolstat == SCIP_NLPSOLSTAT_UNKNOWN) && nnlpcands > 0
      && (divedepth < 10
         || nnlpcands <= startnnlpcands - divedepth/2
         || (divedepth < maxdivedepth && heurdata->nnlpiterations < maxnnlpiterations && objval < searchbound))
      && !SCIPisStopped(scip) )
   {
      SCIP_CALL( SCIPnewProbingNode(scip) );
      divedepth++;

      /* choose variable fixing:
       * - prefer variables that may not be rounded without destroying NLP feasibility:
       *   - of these variables, round least fractional variable in corresponding direction
       * - if all remaining fractional variables may be rounded without destroying NLP feasibility:
       *   - round variable with least increasing objective value
       */
      bestcand = -1;
      bestobjgain = SCIPinfinity(scip);
      bestfrac = SCIP_INVALID;
      bestcandmayrounddown = TRUE;
      bestcandmayroundup = TRUE;
      bestcandroundup = FALSE;

      /* find best candidate variable */
      for( c = 0; c < nnlpcands; ++c )
      {
         var = nlpcands[c];
         mayrounddown = SCIPvarMayRoundDown(var);
         mayroundup = SCIPvarMayRoundUp(var);
         frac = nlpcandsfrac[c];
         obj = SCIPvarGetObj(var);

         if( SCIPisLT(scip, nlpcandssol[c], SCIPvarGetLbLocal(var)) || SCIPisGT(scip, nlpcandssol[c], SCIPvarGetUbLocal(var)) )
            continue;

         if( mayrounddown || mayroundup )
         {
            /* the candidate may be rounded: choose this candidate only, if the best candidate may also be rounded */
            if( bestcandmayrounddown || bestcandmayroundup )
            {
               /* choose rounding direction:
                * - if variable may be rounded in both directions, round corresponding to the fractionality
                * - otherwise, round in the infeasible direction, because feasible direction is tried by rounding
                *   the current fractional solution
                */
               if( mayrounddown && mayroundup )
                  roundup = (frac > 0.5);
               else
                  roundup = mayrounddown;

               if( roundup )
               {
                  frac = 1.0 - frac;
                  objgain = frac*obj;
               }
               else
                  objgain = -frac*obj;

               /* penalize too small fractions */
               if( frac < 0.01 )
                  objgain *= 1000.0;

               /* prefer decisions on binary variables */
               if( !SCIPvarIsBinary(var) )
                  objgain *= 1000.0;

               /* prefer decisions on cover variables */
               if( covercomputed && heurdata->prefercover && SCIPhashmapExists(varincover, var) )
                  objgain *= 1000.0;

               /* check, if candidate is new best candidate */
               if( SCIPisLT(scip, objgain, bestobjgain) || (SCIPisEQ(scip, objgain, bestobjgain) && frac < bestfrac) )
               {
                  bestcand = c;
                  bestobjgain = objgain;
                  bestfrac = frac;
                  bestcandmayrounddown = mayrounddown;
                  bestcandmayroundup = mayroundup;
                  bestcandroundup = roundup;
               }
            }
         }
         else
         {
            /* the candidate may not be rounded */
            if( frac < 0.5 )
               roundup = FALSE;
            else
            {
               roundup = TRUE;
               frac = 1.0 - frac;
            }

            /* penalize too small fractions */
            if( frac < 0.01 )
               frac += 10.0;

            /* prefer decisions on binary variables */
            if( !SCIPvarIsBinary(var) )
               frac *= 1000.0;

            /* prefer decisions on cover variables */
            if( covercomputed && heurdata->prefercover && SCIPhashmapExists(varincover, var) )
               frac *= 1000.0;

            /* check, if candidate is new best candidate: prefer unroundable candidates in any case */
            if( bestcandmayrounddown || bestcandmayroundup || frac < bestfrac )
            {
               bestcand = c;
               bestfrac = frac;
               bestcandmayrounddown = FALSE;
               bestcandmayroundup = FALSE;
               bestcandroundup = roundup;
            }
            assert(bestfrac < SCIP_INVALID);
         }
      }
      assert(bestcand != -1);

      /* if all candidates are roundable, try to round the solution */
      if( bestcandmayrounddown || bestcandmayroundup )
      {
         SCIP_Bool success;

         /* create solution from diving NLP and try to round it */
         SCIP_CALL( SCIPlinkNLPSol(scip, heurdata->sol) );
         SCIP_CALL( SCIProundSol(scip, heurdata->sol, &success) );

         if( success )
         {
            SCIPdebugMessage("nlpfracdiving found roundable primal solution: obj=%g\n", SCIPgetSolOrigObj(scip, heurdata->sol));

            /* try to add solution to SCIP */
            SCIP_CALL( SCIPtrySol(scip, heurdata->sol, TRUE, FALSE, FALSE, TRUE, &success) );

            /* check, if solution was feasible and good enough */
            if( success )
            {
               SCIPdebugMessage(" -> solution was feasible and good enough\n");
               *result = SCIP_FOUNDSOL;
            }
         }
      }

      var = nlpcands[bestcand];

      backtracked = FALSE;
      do
      {
         /* if the variable is already fixed, numerical troubles may have occurred or
          * variable was fixed by propagation while backtracking => Abort diving!
          */
         if( SCIPvarGetLbLocal(var) >= SCIPvarGetUbLocal(var) - 0.5 )
         {
            SCIPdebugMessage("Selected variable <%s> already fixed to [%g,%g] (solval: %.9f), diving aborted \n",
               SCIPvarGetName(var), SCIPvarGetLbLocal(var), SCIPvarGetUbLocal(var), nlpcandssol[bestcand]);
            cutoff = TRUE;
            break;
         }

         if( backtracked && backtrackdepth > 0 )
         {
            /* round backtrack variable up or down */
            if( backtrackroundup )
            {
               SCIPdebugMessage("  dive %d/%d, NLP iter %"SCIP_LONGINT_FORMAT"/%"SCIP_LONGINT_FORMAT": var <%s>, sol=%g, oldbounds=[%g,%g], newbounds=[%g,%g]\n",
                  divedepth, maxdivedepth, heurdata->nnlpiterations, maxnnlpiterations,
                  SCIPvarGetName(backtrackvar), backtrackvarval, SCIPvarGetLbLocal(backtrackvar), SCIPvarGetUbLocal(backtrackvar),
                  SCIPfeasCeil(scip, backtrackvarval), SCIPvarGetUbLocal(backtrackvar));
               SCIP_CALL( SCIPchgVarLbProbing(scip, backtrackvar, SCIPfeasCeil(scip, backtrackvarval)) );
            }
            else
            {
               SCIPdebugMessage("  dive %d/%d, NLP iter %"SCIP_LONGINT_FORMAT"/%"SCIP_LONGINT_FORMAT": var <%s>, sol=%g, oldbounds=[%g,%g], newbounds=[%g,%g]\n",
                  divedepth, maxdivedepth, heurdata->nnlpiterations, maxnnlpiterations,
                  SCIPvarGetName(backtrackvar), backtrackvarval, SCIPvarGetLbLocal(backtrackvar), SCIPvarGetUbLocal(backtrackvar),
                  SCIPvarGetLbLocal(backtrackvar), SCIPfeasFloor(scip, backtrackvarval));
               SCIP_CALL( SCIPchgVarUbProbing(scip, backtrackvar, SCIPfeasFloor(scip, backtrackvarval)) );
            }
            /* forget about backtrack variable */
            backtrackdepth = -1;
         }
         else
         {
            /* apply rounding of best candidate */
            if( bestcandroundup == !backtracked )
            {
               /* round variable up */
               SCIPdebugMessage("  dive %d/%d, NLP iter %"SCIP_LONGINT_FORMAT"/%"SCIP_LONGINT_FORMAT": var <%s>, round=%u/%u, sol=%g, oldbounds=[%g,%g], newbounds=[%g,%g]\n",
                  divedepth, maxdivedepth, heurdata->nnlpiterations, maxnnlpiterations,
                  SCIPvarGetName(var), bestcandmayrounddown, bestcandmayroundup,
                  nlpcandssol[bestcand], SCIPvarGetLbLocal(var), SCIPvarGetUbLocal(var),
                  SCIPfeasCeil(scip, nlpcandssol[bestcand]), SCIPvarGetUbLocal(var));
               SCIP_CALL( SCIPchgVarLbProbing(scip, var, SCIPfeasCeil(scip, nlpcandssol[bestcand])) );

               /* remember variable for backtracking, if we have none yet (e.g., we are just after NLP solve) or we are half way to the next NLP solve */
               if( backtrackdepth == -1 || (divedepth - lastnlpsolvedepth == (int)(MIN(heurdata->fixquot * nnlpcands, nlpbranchcands)/2.0)) )
               {
                  backtrackdepth   = divedepth;
                  backtrackvar     = var;
                  backtrackvarval  = nlpcandssol[bestcand];
                  backtrackroundup = FALSE;
               }
            }
            else
            {
               /* round variable down */
               SCIPdebugMessage("  dive %d/%d, NLP iter %"SCIP_LONGINT_FORMAT"/%"SCIP_LONGINT_FORMAT": var <%s>, round=%u/%u, sol=%g, oldbounds=[%g,%g], newbounds=[%g,%g]\n",
                  divedepth, maxdivedepth, heurdata->nnlpiterations, maxnnlpiterations,
                  SCIPvarGetName(var), bestcandmayrounddown, bestcandmayroundup,
                  nlpcandssol[bestcand], SCIPvarGetLbLocal(var), SCIPvarGetUbLocal(var),
                  SCIPvarGetLbLocal(var), SCIPfeasFloor(scip, nlpcandssol[bestcand]));
               SCIP_CALL( SCIPchgVarUbProbing(scip, var, SCIPfeasFloor(scip, nlpcandssol[bestcand])) );

               /* remember variable for backtracking, if we have none yet (e.g., we are just after NLP solve) or we are half way to the next NLP solve */
               if( backtrackdepth == -1 || (divedepth - lastnlpsolvedepth == (int)(MIN(heurdata->fixquot * nnlpcands, nlpbranchcands)/2.0)) )
               {
                  backtrackdepth   = divedepth;
                  backtrackvar     = var;
                  backtrackvarval  = nlpcandssol[bestcand];
                  backtrackroundup = TRUE;
               }
            }
         }

         /* apply domain propagation */
         SCIP_CALL( SCIPpropagateProbing(scip, 0, &cutoff, NULL) );

         solvenlp = FALSE;
         if( !cutoff )
         {
            solvenlp = (lastnlpsolvedepth < divedepth - MIN(heurdata->fixquot * nnlpcands, nlpbranchcands));
            if( !solvenlp )
            {
               /* check if fractional NLP variables are left (some may have been fixed by propagation) */
               for( c = 0; c < nnlpcands; ++c )
               {
                  if( SCIPisLT(scip, nlpcandssol[c], SCIPvarGetLbLocal(var)) || SCIPisGT(scip, nlpcandssol[c], SCIPvarGetUbLocal(var)) )
                     continue;
                  else
                     break;
               }
               if( c == nnlpcands )
                  solvenlp = TRUE;
            }
         }
         else
         {
            SCIPdebugMessage("  *** cutoff detected in propagation at level %d\n", SCIPgetProbingDepth(scip));
         }

         nlpsolstat = SCIP_NLPSOLSTAT_UNKNOWN;

         /* if all variables in the cover are fixed or there is no fractional variable in the cover, 
          * then solve a sub-MIP 
          */
         if( !cutoff && solvesubmip && covercomputed && 
            (heurdata->nfixedcovervars == ncovervars || 
               (heurdata->nfixedcovervars >= (ncovervars+1)/2 && !SCIPhashmapExists(varincover, var))) )
         {
            int probingdepth;

            solvesubmip = FALSE;
            probingdepth = SCIPgetProbingDepth(scip);
            assert(probingdepth >= 1);

            if( heurdata->nfixedcovervars != ncovervars )
            {
               /* fix all remaining cover variables */
               for( c = 0; c < ncovervars && !cutoff ; c++ )
               {
                  SCIP_Real lb;
                  SCIP_Real ub;
                  lb = SCIPvarGetLbLocal(covervars[c]);
                  ub = SCIPvarGetUbLocal(covervars[c]);
                  if( !SCIPisFeasEQ(scip, lb, ub) )
                  {
                     SCIP_Real nlpsolval;

                     /* adopt lpsolval w.r.t. intermediate bound changes by propagation */
                     nlpsolval = SCIPvarGetNLPSol(covervars[c]);
                     nlpsolval = MIN(nlpsolval,ub);
                     nlpsolval = MAX(nlpsolval,lb);
                     assert(SCIPvarGetType(covervars[c]) == SCIP_VARTYPE_CONTINUOUS || SCIPisFeasIntegral(scip, nlpsolval));

                     /* fix and propagate */
                     SCIP_CALL( SCIPnewProbingNode(scip) );
                     SCIP_CALL( SCIPchgVarLbProbing(scip, covervars[c], nlpsolval) );
                     SCIP_CALL( SCIPchgVarUbProbing(scip, covervars[c], nlpsolval) );
                     SCIP_CALL( SCIPpropagateProbing(scip, 0, &cutoff, NULL) );
                  }
               }
            }

            /* solve sub-MIP or return to standard diving */
            if( cutoff )
            {
               SCIP_CALL( SCIPbacktrackProbing(scip, probingdepth) );
            }
            else
            {
               SCIP_Bool success;
               success = FALSE;
               
               SCIP_CALL( solveSubMIP(scip, heur, covervars, ncovervars, &success));
               if( success )
                  *result = SCIP_FOUNDSOL;     
               backtracked = TRUE; /* to avoid backtracking */
               nnlpcands = 0; /* to force termination */
               cutoff = TRUE;
            }
         }

         if( !cutoff && solvenlp )
         {
            /* resolve the diving NLP */
            SCIP_NLPTERMSTAT termstat;
            SCIP_NLPSTATISTICS* nlpstatistics;

            /* set iteration limit; @todo reset limit when heuristic finished */
            SCIP_CALL( SCIPsetNLPIntPar(scip, SCIP_NLPPAR_ITLIM, MAX((int)(maxnnlpiterations - heurdata->nnlpiterations), MINNLPITER)) );

            SCIP_CALL( SCIPsolveNLP(scip) );
            STATISTIC( ++heurdata->nnlpsolves; )

            termstat = SCIPgetNLPTermstat(scip);
            if( termstat >= SCIP_NLPTERMSTAT_NUMERR )
            {
               SCIPwarningMessage("Error while solving NLP in Fracdiving heuristic; NLP solve terminated with code <%d>\n", termstat);
               nlperror = TRUE;
               break;
            }

            /* update iteration count */
            SCIP_CALL( SCIPnlpStatisticsCreate(&nlpstatistics) );
            SCIP_CALL( SCIPgetNLPStatistics(scip, nlpstatistics) );
            heurdata->nnlpiterations += SCIPnlpStatisticsGetNIterations(nlpstatistics);
            SCIPnlpStatisticsFree(&nlpstatistics);

            /* get NLP solution status, objective value, and fractional variables, that should be integral */
            nlpsolstat = SCIPgetNLPSolstat(scip);
            cutoff = (termstat == SCIP_NLPTERMSTAT_UOBJLIM || nlpsolstat == SCIP_NLPSOLSTAT_LOCINFEASIBLE || nlpsolstat == SCIP_NLPSOLSTAT_GLOBINFEASIBLE);

            if( cutoff )
            {
               SCIPdebugMessage("  *** cutoff detected in NLP solving at level %d, nlpsolstat: %d\n", SCIPgetProbingDepth(scip), nlpsolstat);
            }
            else
            {
               /* remember that we have solve NLP on this depth successfully */
               lastnlpsolvedepth = divedepth;
               /* forget previous backtrack variable, we will never go back to a depth before the current one */
               backtrackdepth = -1;
            }

            /* resolve LP */  
            if( !cutoff && !lperror && heurdata->preferlpfracs )
            {
               SCIP_CALL( SCIPsolveProbingLP(scip, 100, &lperror) );

               /* get LP solution status, objective value, and fractional variables, that should be integral */
               lpsolstat = SCIPgetLPSolstat(scip);
               cutoff = (lpsolstat == SCIP_LPSOLSTAT_OBJLIMIT || lpsolstat == SCIP_LPSOLSTAT_INFEASIBLE);

               if( lpsolstat == SCIP_LPSOLSTAT_OPTIMAL )
                  nlpbranchcands = SCIPgetNLPBranchCands(scip);
               else
                  nlpbranchcands = 0;
            }
         }

         /* perform backtracking if a cutoff was detected */
         if( cutoff && !backtracked && heurdata->backtrack )
         {
            if( backtrackdepth == -1 )
            {
               /* backtrack one step */
               SCIPdebugMessage("  *** cutoff detected at level %d - backtracking one step\n", SCIPgetProbingDepth(scip));
               SCIP_CALL( SCIPbacktrackProbing(scip, SCIPgetProbingDepth(scip)-1) );
               SCIP_CALL( SCIPnewProbingNode(scip) );
            }
            else
            {
               /* if we have a stored a depth for backtracking, go there */
               SCIPdebugMessage("  *** cutoff detected at level %d - backtracking to depth %d\n", SCIPgetProbingDepth(scip), backtrackdepth);
               SCIP_CALL( SCIPbacktrackProbing(scip, backtrackdepth-1) );
               SCIP_CALL( SCIPnewProbingNode(scip) );
               divedepth = backtrackdepth;
               /* @todo if backtrackdepth is lastnlpsolvedepth-1, reduce fixquot, so we don't wait with NLP solves for too long */
            }
            backtracked = TRUE;
         }
         else
            backtracked = FALSE;
      }
      while( backtracked );

      if( !nlperror && !cutoff && nlpsolstat <= SCIP_NLPSOLSTAT_FEASIBLE )
      {
         /* get new objective value */
         oldobjval = objval;
         objval = SCIPgetNLPObjval(scip);
#if 0
         /* update pseudo cost values */
         if( SCIPisGT(scip, objval, oldobjval) )
         {
            if( bestcandroundup )
            {
               SCIP_CALL( SCIPupdateVarPseudocost(scip, nlpcands[bestcand], 1.0-nlpcandsfrac[bestcand],
                     objval - oldobjval, 1.0) );
            }
            else
            {
               SCIP_CALL( SCIPupdateVarPseudocost(scip, nlpcands[bestcand], 0.0-nlpcandsfrac[bestcand],
                     objval - oldobjval, 1.0) );
            }
         }
#endif
         /* get new fractional variables */
         SCIP_CALL( SCIPgetNLPFracVars(scip, &nlpcands, &nlpcandssol, &nlpcandsfrac, &nnlpcands, NULL) );

         if( heurdata->preferlpfracs && lpsolstat == SCIP_LPSOLSTAT_OPTIMAL )
            for( c = 0; c < nnlpcands; ++c )
            {
               var = nlpcands[c];
               /* prefer decisions on variables which are also fractional in LP solution */
               if( SCIPisFeasIntegral(scip, SCIPgetSolVal(scip, NULL, var)) )
                  nlpcandsfrac[c] *= 100.0;
            }

      }
      SCIPdebugMessage("   -> nlpsolstat=%d, objval=%g/%g, nfrac=%d\n", nlpsolstat, objval, searchbound, nnlpcands);
   }

#if 1
   SCIPdebugMessage("NLP fracdiving ABORT due to ");
   if( nlperror || (nlpsolstat > SCIP_NLPSOLSTAT_LOCINFEASIBLE && nlpsolstat != SCIP_NLPSOLSTAT_UNKNOWN) )
   {
      SCIPdebugPrintf("NLP sucks - nlperror: %d nlpsolstat: %d \n", nlperror, nlpsolstat);
      STATISTIC( heurdata->nfailnlperror++; )
   }
   else if( SCIPisStopped(scip) || cutoff )
   {
      SCIPdebugPrintf("LIMIT hit - stop: %d cutoff: %d \n", SCIPisStopped(scip), cutoff);
      STATISTIC( heurdata->nfailcutoff++; )
   }
   else if(! (divedepth < 10
         || nnlpcands <= startnnlpcands - divedepth/2
         || (divedepth < maxdivedepth && heurdata->nnlpiterations < maxnnlpiterations && objval < searchbound) ) )
   {
      SCIPdebugPrintf("TOO DEEP - divedepth: %4d cands halfed: %d ltmaxdepth: %d ltmaxiter: %d bound: %d\n", divedepth, 
         (nnlpcands > startnnlpcands - divedepth/2), (divedepth >= maxdivedepth), (heurdata->nnlpiterations >= maxnnlpiterations),
         (objval >= searchbound));   
      STATISTIC( heurdata->nfaildepth++; )
   }
   else if ( nnlpcands == 0 && !nlperror && !cutoff && nlpsolstat <= SCIP_NLPSOLSTAT_FEASIBLE )
   {
      SCIPdebugPrintf("SUCCESS\n");
   }
   else
   {
      SCIPdebugPrintf("UNKNOWN, very mysterical reason\n");
   }
#endif

   /* check if a solution has been found */
   if( nnlpcands == 0 && !nlperror && !cutoff && nlpsolstat <= SCIP_NLPSOLSTAT_FEASIBLE )
   {
      SCIP_Bool success;

      /* create solution from diving NLP */
      SCIP_CALL( SCIPlinkNLPSol(scip, heurdata->sol) );
      SCIPdebugMessage("nlpfracdiving found primal solution: obj=%g\n", SCIPgetSolOrigObj(scip, heurdata->sol));

      /* try to add solution to SCIP */
#ifdef SCIP_DEBUG
      SCIP_CALL( SCIPtrySol(scip, heurdata->sol, TRUE, FALSE, FALSE, TRUE, &success) );
#else
      SCIP_CALL( SCIPtrySol(scip, heurdata->sol, FALSE, FALSE, FALSE, TRUE, &success) );
#endif

      /* check, if solution was feasible and good enough */
      if( success )
      {
         SCIPdebugMessage(" -> solution was feasible and good enough\n");
         *result = SCIP_FOUNDSOL;
      }
      else
      {
         SCIPdebugMessage(" -> solution was not accepted\n");
      }
   }

   /* end diving */
   SCIP_CALL( SCIPendProbing(scip) );

   /* free hash map and drop variable bound change events */
   if( covercomputed && heurdata->solvesubmip)
   {
      assert(heurdata->eventhdlr != NULL);
      assert(heurdata->nfixedcovervars == 0);
      assert(varincover != NULL);

      SCIPhashmapFree(&varincover);

      /* drop bound change events of cover variables */
      for( c = 0; c < ncovervars; c++ )
      {
         SCIP_CALL( SCIPdropVarEvent(scip, covervars[c], SCIP_EVENTTYPE_BOUNDCHANGED, heurdata->eventhdlr, (SCIP_EVENTDATA*)heurdata, -1) );
      }
   }
   else 
      assert(varincover == NULL);

   /* free array of cover variables */
   if( heurdata->prefercover || heurdata->solvesubmip )
   {
      assert(covervars != NULL);
      SCIPfreeBufferArray(scip, &covervars);
   }
   else 
      assert(covervars == NULL);

   if( *result == SCIP_FOUNDSOL )
      heurdata->nsuccess++;

   SCIPdebugMessage("nlpfracdiving heuristic finished\n");

   return SCIP_OKAY;
}




/*
 * heuristic specific interface methods
 */

/** creates the fracdiving heuristic and includes it in SCIP */
SCIP_RETCODE SCIPincludeHeurNlpFracdiving(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_HEURDATA* heurdata;

   /* create heuristic data */
   SCIP_CALL( SCIPallocMemory(scip, &heurdata) );

   /* include heuristic */
   SCIP_CALL( SCIPincludeHeur(scip, HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ, HEUR_FREQOFS,
         HEUR_MAXDEPTH, HEUR_TIMING, HEUR_USESSUBSCIP,
         heurCopyNlpFracdiving,
         heurFreeNlpFracdiving, heurInitNlpFracdiving, heurExitNlpFracdiving,
         heurInitsolNlpFracdiving, heurExitsolNlpFracdiving, heurExecNlpFracdiving,
         heurdata) );

   /* create event handler for bound change events */
   SCIP_CALL( SCIPincludeEventhdlr(scip, EVENTHDLR_NAME, EVENTHDLR_DESC,
         NULL,NULL, NULL, NULL, NULL, NULL, NULL, eventExecNlpFracdiving, NULL) );

   /* get event handler for bound change events */
   heurdata->eventhdlr = SCIPfindEventhdlr(scip, EVENTHDLR_NAME);
   if ( heurdata->eventhdlr == NULL )
   {
      SCIPerrorMessage("event handler for "HEUR_NAME" heuristic not found.\n");
      return SCIP_PLUGINNOTFOUND;
   }

   /* fracdiving heuristic parameters */
   SCIP_CALL( SCIPaddRealParam(scip,
         "heuristics/"HEUR_NAME"/minreldepth",
         "minimal relative depth to start diving",
         &heurdata->minreldepth, TRUE, DEFAULT_MINRELDEPTH, 0.0, 1.0, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip,
         "heuristics/"HEUR_NAME"/maxreldepth",
         "maximal relative depth to start diving",
         &heurdata->maxreldepth, TRUE, DEFAULT_MAXRELDEPTH, 0.0, 1.0, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip,
         "heuristics/"HEUR_NAME"/maxnlpiterquot",
         "maximal fraction of diving LP iterations compared to node LP iterations",
         &heurdata->maxnlpiterquot, FALSE, DEFAULT_MAXNLPITERQUOT, 0.0, SCIP_REAL_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddIntParam(scip,
         "heuristics/"HEUR_NAME"/maxnlpiterofs",
         "additional number of allowed LP iterations",
         &heurdata->maxnlpiterofs, FALSE, DEFAULT_MAXNLPITEROFS, 0, INT_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip,
         "heuristics/"HEUR_NAME"/maxdiveubquot",
         "maximal quotient (curlowerbound - lowerbound)/(cutoffbound - lowerbound) where diving is performed (0.0: no limit)",
         &heurdata->maxdiveubquot, TRUE, DEFAULT_MAXDIVEUBQUOT, 0.0, 1.0, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip,
         "heuristics/"HEUR_NAME"/maxdiveavgquot",
         "maximal quotient (curlowerbound - lowerbound)/(avglowerbound - lowerbound) where diving is performed (0.0: no limit)",
         &heurdata->maxdiveavgquot, TRUE, DEFAULT_MAXDIVEAVGQUOT, 0.0, SCIP_REAL_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip,
         "heuristics/"HEUR_NAME"/maxdiveubquotnosol",
         "maximal UBQUOT when no solution was found yet (0.0: no limit)",
         &heurdata->maxdiveubquotnosol, TRUE, DEFAULT_MAXDIVEUBQUOTNOSOL, 0.0, 1.0, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip,
         "heuristics/"HEUR_NAME"/maxdiveavgquotnosol",
         "maximal AVGQUOT when no solution was found yet (0.0: no limit)",
         &heurdata->maxdiveavgquotnosol, TRUE, DEFAULT_MAXDIVEAVGQUOTNOSOL, 0.0, SCIP_REAL_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip,
         "heuristics/"HEUR_NAME"/backtrack",
         "use one level of backtracking if infeasibility is encountered?",
         &heurdata->backtrack, FALSE, DEFAULT_BACKTRACK, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip,
         "heuristics/"HEUR_NAME"/preferlpfracs",
         "prefer variables that are also fractional in LP solution?",
         &heurdata->preferlpfracs, TRUE, DEFAULT_PREFERLPFRACS, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip,
         "heuristics/"HEUR_NAME"/minsuccquot",
         "heuristic will not run if less then this percentage of calls succeeded (0.0: no limit)",
         &heurdata->minsuccquot, FALSE, DEFAULT_MINSUCCQUOT, 0.0, 1.0, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip,
         "heuristics/"HEUR_NAME"/fixquot",
         "percentage of fractional variables that should be fixed before the next NLP solve",
         &heurdata->fixquot, FALSE, DEFAULT_FIXQUOT, 0.0, 1.0, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip,
         "heuristics/"HEUR_NAME"/prefercover",
         "should variables in a minimal cover be preferred?",
         &heurdata->prefercover, FALSE, DEFAULT_PREFERCOVER, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip,
         "heuristics/"HEUR_NAME"/solvesubmip",
         "should a sub-MIP be solved if all cover variables are fixed?",
         &heurdata->solvesubmip, FALSE, DEFAULT_SOLVESUBMIP, NULL, NULL) );

   return SCIP_OKAY;
}

