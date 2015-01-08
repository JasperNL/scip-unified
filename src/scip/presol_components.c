/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2015 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
//#define SCIP_DEBUG
//#define SCIP_MORE_DEBUG
/**@file   presol_components.c
 * @ingroup PRESOLVERS
 * @brief  solve independent components in advance
 * @author Dieter Weninger
 * @author Gerald Gamrath
 *
 * This presolver looks for independent components at the end of the presolving.
 * If independent components are found in which a maximum number of discrete variables
 * is not exceeded, the presolver tries to solve them in advance as subproblems.
 * Afterwards, if a subproblem was solved to optimality, the corresponding
 * variables/constraints can be fixed/deleted in the main problem.
 *
 * @todo simulation of presolving without solve
 * @todo solve all components with less than given size, count number of components with nodelimit reached;
 *       if all components could be solved within nodelimit (or all but x), continue solving components in
 *       increasing order until one hit the node limit
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>

#include "scip/presol_components.h"

#define PRESOL_NAME            "components"
#define PRESOL_DESC            "components presolver"
#define PRESOL_PRIORITY        -9200000      /**< priority of the presolver (>= 0: before, < 0: after constraint handlers); combined with propagators */
#define PRESOL_MAXROUNDS             -1      /**< maximal number of presolving rounds the presolver participates in (-1: no limit) */
#define PRESOL_TIMING           SCIP_PRESOLTIMING_EXHAUSTIVE /* timing of the presolver (fast, medium, or exhaustive) */

#define DEFAULT_WRITEPROBLEMS     FALSE      /**< should the single components be written as an .cip-file? */
#define DEFAULT_MAXINTVARS          500      /**< maximum number of integer (or binary) variables to solve a subproblem directly (-1: no solving) */
#define DEFAULT_NODELIMIT       10000LL      /**< maximum number of nodes to be solved in subproblems */
#define DEFAULT_INTFACTOR           1.0      /**< the weight of an integer variable compared to binary variables */
#define DEFAULT_RELDECREASE         0.2      /**< percentage by which the number of variables has to be decreased after the last component solving
                                              *   to allow running again (1.0: do not run again) */
#define DEFAULT_FEASTOLFACTOR       1.0      /**< default value for parameter to increase the feasibility tolerance in all sub-SCIPs */

#ifdef SCIP_STATISTIC
static int NCATEGORIES = 6;
static int CATLIMITS[] = {0,20,50,100,500};
#endif

/*
 * Data structures
 */

/** control parameters */
struct SCIP_PresolData
{
   SCIP*                 subscip;            /** sub-SCIP used to solve single components */
   SCIP_Longint          nodelimit;          /** maximum number of nodes to be solved in subproblems */
   SCIP_Real             intfactor;          /** the weight of an integer variable compared to binary variables */
   SCIP_Real             reldecrease;        /** percentage by which the number of variables has to be decreased after the last component solving
                                              *  to allow running again (1.0: do not run again) */
   SCIP_Real             feastolfactor;      /** parameter to increase the feasibility tolerance in all sub-SCIPs */
   SCIP_Bool             didsearch;          /** did the presolver already search for components? */
   SCIP_Bool             pluginscopied;      /** was the copying of the plugins successful? */
   SCIP_Bool             writeproblems;      /** should the single components be written as an .cip-file? */
   int                   maxintvars;         /** maximum number of integer (or binary) variables to solve a subproblem directly (-1: no solving) */
   int                   lastnvars;          /** number of variables after last run of the presolver */

   SCIP_SOL*             bestsol;
   SCIP**                components;
   SCIP_VAR***           compvars;
   SCIP_HASHMAP**        varmaps;
   SCIP_Real*            lastdualbound;
   SCIP_Real*            lastprimalbound;
   SCIP_Longint*         lastnodelimit;
   SCIP_STATUS*          laststatus;
   int*                  lastround;
   int*                  lastsolindex;
   int*                  ncompvars;
   int                   ncomponents;
   int                   nfeascomps;
   int                   nsolvedcomps;
#ifdef SCIP_STATISTIC
   int*                  compspercat;        /** number of components of the different categories */
   SCIP_Real             subsolvetime;       /** total solving time of the subproblems */
   int                   nsinglevars;        /** number of components with a single variable without constraint */
#endif
};

/*
 * Statistic methods
 */

#ifdef SCIP_STATISTIC
/** initialize data for statistics */
static
SCIP_RETCODE initStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOLDATA*      presoldata          /**< presolver data */
   )
{
   assert(scip != NULL);
   assert(presoldata != NULL);

   SCIP_CALL( SCIPallocMemoryArray(scip, &presoldata->compspercat, NCATEGORIES) );
   BMSclearMemoryArray(presoldata->compspercat, NCATEGORIES);

   presoldata->nsinglevars = 0;
   presoldata->subsolvetime = 0.0;

   return SCIP_OKAY;
}

/** free data for statistics */
static
void freeStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOLDATA*      presoldata          /**< presolver data */
   )
{
   assert(scip != NULL);
   assert(presoldata != NULL);

   SCIPfreeMemoryArray(scip, &presoldata->compspercat);
}

/** reset data for statistics */
static
void resetStatistics(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOLDATA*      presoldata          /**< presolver data */
   )
{
   assert(scip != NULL);
   assert(presoldata != NULL);

   BMSclearMemoryArray(presoldata->compspercat, NCATEGORIES);

   presoldata->nsinglevars = 0;
   presoldata->subsolvetime = 0.0;
}


/** statistics: categorize the component with the given number of binary and integer variables */
static
void updateStatisticsComp(
   SCIP_PRESOLDATA*      presoldata,         /**< presolver data */
   int                   nbinvars,           /**< number of binary variables */
   int                   nintvars            /**< number of integer variables */
   )
{
   int ndiscretevars;
   int i;

   assert(presoldata != NULL);

   ndiscretevars = nbinvars + nintvars;

   /* check into which category the component belongs by looking at the number of discrete variables */
   for( i = 0; i < (NCATEGORIES - 1); ++i )
   {
      if( ndiscretevars <= CATLIMITS[i] )
      {
         presoldata->compspercat[i]++;
         break;
      }
   }

   /* number of discrete variables greater than all limits, so component belongs to last category */
   if( i == (NCATEGORIES - 1) )
      presoldata->compspercat[i]++;
}

/** statistics: increase the number of components with a single variable and no constraints */
static
void updateStatisticsSingleVar(
   SCIP_PRESOLDATA*      presoldata          /**< presolver data */
   )
{
   assert(presoldata != NULL);

   presoldata->nsinglevars++;
}

/** statistics: update the total subproblem solving time */
static
void updateStatisticsSubsolvetime(
   SCIP_PRESOLDATA*      presoldata,         /**< presolver data */
   SCIP_Real             subsolvetime        /**< subproblem solving time to add to the statistics */
   )
{
   assert(presoldata != NULL);

   presoldata->subsolvetime += subsolvetime;
}

/** print statistics */
static
void printStatistics(
   SCIP_PRESOLDATA*      presoldata          /**< presolver data */
   )
{
   int i;

   assert(presoldata != NULL);

   printf("############\n");
   printf("# Connected Components Presolver Statistics:\n");

   printf("# Categorization:");
   for( i = 0; i < NCATEGORIES - 1; ++i )
   {
      printf("[<= %d: %d]", CATLIMITS[i], presoldata->compspercat[i]);
   }
   printf("[> %d: %d]\n", CATLIMITS[NCATEGORIES - 2], presoldata->compspercat[NCATEGORIES - 1]);
   printf("# Components without constraints: %d\n", presoldata->nsinglevars);
   printf("# Total subproblem solving time: %.2f\n", presoldata->subsolvetime);
   printf("############\n");
}
#endif


/*
 * Local methods
 */

/** (continues) solving a connected component */
static
SCIP_RETCODE solveComponent(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOLDATA*      presoldata,         /**< presolver data */
   int                   comp,               /**< number of the component */
   SCIP_Longint          nodelimit,          /**< node limit for the solving */
   SCIP_Real             gaplimit,           /**< gap limit for the solving */
   int*                  nsolvedprobs,       /**< pointer to increase, if the subproblem was solved */
   SCIP_RESULT*          result              /**< pointer to store the result of the solving process */
   )
{
   SCIP* subscip;
   SCIP_Real timelimit;
   SCIP_Real softtimelimit;
   SCIP_Real memorylimit;
   SCIP_STATUS status;

   assert(scip != NULL);
   assert(presoldata != NULL);
   assert(nsolvedprobs != NULL);

   *result = SCIP_DIDNOTRUN;

#ifndef SCIP_DEBUG
   SCIPverbMessage(scip, SCIP_VERBLEVEL_FULL, NULL, "solve sub-SCIP for component %d\n", comp);
#else
   SCIPdebugMessage("solve sub-SCIP for component %d\n", comp);
#endif

   subscip = presoldata->components[comp];

   /* check whether there is enough time and memory left */
   SCIP_CALL( SCIPgetRealParam(scip, "limits/time", &timelimit) );
   if( !SCIPisInfinity(scip, timelimit) )
      timelimit -= SCIPgetSolvingTime(scip);
   timelimit += SCIPgetSolvingTime(subscip);

   /* check whether there is enough time and memory left */
   SCIP_CALL( SCIPgetRealParam(scip, "limits/softtime", &softtimelimit) );

   if( softtimelimit > -0.5 )
   {
      softtimelimit -= SCIPgetSolvingTime(scip);
      softtimelimit += SCIPgetSolvingTime(subscip);
      softtimelimit = MAX(softtimelimit, 0.0);
   }

   SCIP_CALL( SCIPgetRealParam(scip, "limits/memory", &memorylimit) );

   /* substract the memory already used by the main SCIP and the estimated memory usage of external software */
   if( !SCIPisInfinity(scip, memorylimit) )
   {
      memorylimit -= SCIPgetMemUsed(scip)/1048576.0;
      memorylimit -= SCIPgetMemExternEstim(scip)/1048576.0;
   }

   /* abort if no time is left or not enough memory to create a copy of SCIP, including external memory usage */
   // todo memory limit
   if( timelimit <= 0.0 )
   {
      SCIPdebugMessage("--> not solved (not enough memory or time left)\n");
      return SCIP_OKAY;
   }

   /* set gap limit */
   SCIP_CALL( SCIPsetRealParam(subscip, "limits/gap", gaplimit) );

   /* set time and memory limit for the subproblem */
   SCIP_CALL( SCIPsetRealParam(subscip, "limits/time", timelimit) );
   SCIP_CALL( SCIPsetRealParam(subscip, "limits/softtime", softtimelimit) );
   SCIP_CALL( SCIPsetRealParam(subscip, "limits/memory", memorylimit) );

   SCIP_CALL( SCIPsetLongintParam(subscip, "limits/nodes", nodelimit) );
   presoldata->lastnodelimit[comp] = nodelimit;

   if( *nsolvedprobs == presoldata->ncomponents - 1 )
   {
      SCIP_CALL( SCIPsetIntParam(subscip, "display/verblevel", SCIP_VERBLEVEL_HIGH) );
   }

   /* solve the subproblem */
   SCIP_CALL( SCIPsolve(subscip) );

#ifdef SCIP_MORE_DEBUG
   SCIP_CALL( SCIPprintStatistics(subscip, NULL) );
#endif

   SCIPstatistic( updateStatisticsSubsolvetime(presoldata, SCIPgetSolvingTime(subscip)) );

   status = SCIPgetStatus(subscip);

   switch( status )
   {
   case SCIP_STATUS_INFEASIBLE:
#ifndef SCIP_DEBUG
      SCIPverbMessage(scip, SCIP_VERBLEVEL_FULL, NULL, "--> infeasible (status=%d, time=%.2f)\n",
         SCIPgetStatus(subscip), SCIPgetSolvingTime(subscip));
#else
      SCIPdebugMessage("--> infeasible (status=%d, time=%.2f)\n",
         SCIPgetStatus(subscip), SCIPgetSolvingTime(subscip));
#endif
      *result = SCIP_CUTOFF;
      break;
   case SCIP_STATUS_UNBOUNDED:
   case SCIP_STATUS_INFORUNBD:
#ifndef SCIP_DEBUG
      SCIPverbMessage(scip, SCIP_VERBLEVEL_FULL, NULL, "--> unbounded (status=%d, time=%.2f)\n",
         SCIPgetStatus(subscip), SCIPgetSolvingTime(subscip));
#else
      SCIPdebugMessage("--> unbounded (status=%d, time=%.2f)\n",
         SCIPgetStatus(subscip), SCIPgetSolvingTime(subscip));
#endif
      /* TODO: store unbounded ray in original SCIP data structure */
      *result = SCIP_UNBOUNDED;
      break;
   default:
   {
      SCIP_SOL* sol = SCIPgetBestSol(subscip);
      SCIP_VAR* var;
      SCIP_VAR* subvar;
      int v;

#ifndef SCIP_DEBUG
      SCIPverbMessage(scip, SCIP_VERBLEVEL_FULL, NULL, "--> (status=%d, time=%.2f): gap: %12.5g%% absgap: %16.9g\n",
         SCIPgetStatus(subscip), SCIPgetSolvingTime(subscip), 100.0*SCIPgetGap(subscip), SCIPgetPrimalbound(subscip) - SCIPgetDualbound(subscip));
#else
      SCIPdebugMessage("--> (status=%d, time=%.2f): gap: %12.5g%% absgap: %16.9g\n",
         SCIPgetStatus(subscip), SCIPgetSolvingTime(subscip), 100.0*SCIPgetGap(subscip), SCIPgetPrimalbound(subscip) - SCIPgetDualbound(subscip));
#endif

      if( sol != NULL && presoldata->lastsolindex[comp] != SCIPsolGetIndex(sol) )
      {
         presoldata->lastsolindex[comp] = SCIPsolGetIndex(sol);

         if( SCIPisInfinity(scip, presoldata->lastprimalbound[comp]) )
         {
            ++(presoldata->nfeascomps);
         }

         /* store primal solution values */
         for( v = 0; v < presoldata->ncompvars[comp]; ++v )
         {
            var = presoldata->compvars[comp][v];
            assert(var != NULL);

            subvar = (SCIP_VAR*)SCIPhashmapGetImage(presoldata->varmaps[comp], (void*)var);
            assert(subvar != NULL);

            SCIP_CALL( SCIPsetSolVal(scip, presoldata->bestsol, var,
                  SCIPgetSolVal(presoldata->components[comp], sol, subvar)) );
         }

         /* we have a feasible solution for each component */
         if( presoldata->nfeascomps == presoldata->ncomponents )
         {
            SCIP_Bool feasible;

            SCIP_CALL( SCIPcheckSolOrig(scip, presoldata->bestsol, &feasible, TRUE, TRUE) );

            printf("feasible = %d, obj = %g\n", feasible, SCIPgetSolOrigObj(scip, sol));

            SCIP_CALL( SCIPaddSol(scip, presoldata->bestsol, &feasible) );
         }
      }
   }
   }

   presoldata->lastdualbound[comp] = SCIPgetDualbound(subscip);
   presoldata->lastprimalbound[comp] = SCIPgetPrimalbound(subscip);
   presoldata->laststatus[comp] = SCIPgetStatus(subscip);

   if( status == SCIP_STATUS_OPTIMAL )
   {
      ++(presoldata->nsolvedcomps);
#ifndef SCIP_DEBUG
      SCIPverbMessage(scip, SCIP_VERBLEVEL_FULL, NULL, "--> solved to optimality (status=%d, time=%.2f)\n",
         SCIPgetStatus(subscip), SCIPgetSolvingTime(subscip));
#else
      SCIPdebugMessage("--> solved to optimality (status=%d, time=%.2f)\n",
         SCIPgetStatus(subscip), SCIPgetSolvingTime(subscip));
#endif

      /* free component */
      SCIP_CALL( SCIPfree(&subscip) );
      presoldata->components[comp] = NULL;
   }


   return SCIP_OKAY;
}

/** copies a connected component consisting of the given constraints and variables into a sub-SCIP
 *  and tries to solve the sub-SCIP to optimality
 */
static
SCIP_RETCODE copyComponent(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOLDATA*      presoldata,         /**< presolver data */
   SCIP_HASHMAP*         consmap,            /**< constraint hashmap used to improve performance */
   int                   compnr,             /**< number of the component */
   SCIP_CONS**           conss,              /**< constraints contained in this component */
   int                   nconss,             /**< number of constraints contained in this component */
   SCIP_VAR**            vars,               /**< variables contained in this component */
   SCIP_Real*            fixvals,            /**< array to store the values to fix the variables to */
   int                   nvars,              /**< number of variables contained in this component */
   int                   nbinvars,           /**< number of binary variables contained in this component */
   int                   nintvars,           /**< number of integer variables contained in this component */
   int*                  nsolvedprobs,       /**< pointer to increase, if the subproblem was solved */
   int*                  ndeletedvars,       /**< pointer to increase by the number of deleted variables */
   int*                  ndeletedconss,      /**< pointer to increase by the number of deleted constraints */
   SCIP_RESULT*          result              /**< pointer to store the result of the presolving call */
   )
{
   char name[SCIP_MAXSTRLEN];
   SCIP_HASHMAP* varmap;
   SCIP* subscip;
   SCIP_CONS* newcons;
   SCIP_Real feastol;
   SCIP_Bool success;
   int i;

   assert(scip != NULL);
   assert(presoldata != NULL);
   assert(conss != NULL);
   assert(nconss > 0);
   assert(vars != NULL);
   assert(nvars > 0);
   assert(nsolvedprobs != NULL);
   assert(presoldata->ncomponents == compnr);

   *result = SCIP_DIDNOTRUN;

#ifndef SCIP_DEBUG
   SCIPverbMessage(scip, SCIP_VERBLEVEL_FULL, NULL, "build sub-SCIP for component %d: %d vars (%d bin, %d int, %d cont), %d conss\n",
      compnr, nvars, nbinvars, nintvars, nvars - nintvars - nbinvars, nconss);
#else
   SCIPdebugMessage("build sub-SCIP for component %d: %d vars (%d bin, %d int, %d cont), %d conss\n",
      compnr, nvars, nbinvars, nintvars, nvars - nintvars - nbinvars, nconss);
#endif

   SCIP_CALL( SCIPcreate(&presoldata->components[compnr]) );
   subscip = presoldata->components[compnr];
   ++(presoldata->ncomponents);

   /* copy plugins, we omit pricers (because we do not run if there are active pricers) and dialogs */
   success = TRUE;
   SCIP_CALL( SCIPcopyPlugins(scip, subscip, TRUE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE,
         TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, TRUE, &success) );

   /* abort if the plugins were not successfully copied */
   // @todo what should we do here?
   if( !success )
   {
      SCIP_CALL( SCIPfree(&subscip) );
      presoldata->components[compnr] = NULL;
      presoldata->pluginscopied = FALSE;

      // only for debugging
      assert(0);

      return SCIP_OKAY;
   }
   /* copy parameter settings */
   SCIP_CALL( SCIPcopyParamSettings(scip, subscip) );

   if( !SCIPisParamFixed(subscip, "limits/solutions") )
   {
      SCIP_CALL( SCIPsetIntParam(subscip, "limits/solutions", -1) );
   }
   if( !SCIPisParamFixed(subscip, "limits/bestsol") )
   {
      SCIP_CALL( SCIPsetIntParam(subscip, "limits/bestsol", -1) );
   }

   /* reduce the effort spent for hash tables */
   if( !SCIPisParamFixed(subscip, "misc/usevartable") )
   {
      SCIP_CALL( SCIPsetBoolParam(subscip, "misc/usevartable", FALSE) );
   }
   if( !SCIPisParamFixed(subscip, "misc/useconstable") )
   {
      SCIP_CALL( SCIPsetBoolParam(subscip, "misc/useconstable", FALSE) );
   }
   if( !SCIPisParamFixed(subscip, "misc/usesmalltables") )
   {
      SCIP_CALL( SCIPsetBoolParam(subscip, "misc/usesmalltables", TRUE) );
   }

   /* increase feasibility tolerance if asked for */
   if( !SCIPisEQ(scip, presoldata->feastolfactor, 1.0) )
   {
      SCIP_CALL( SCIPgetRealParam(scip, "numerics/feastol", &feastol) );
      SCIP_CALL( SCIPsetRealParam(subscip, "numerics/feastol", feastol * presoldata->feastolfactor) );
   }

   //SCIP_CALL( SCIPsetHeuristics(scip, SCIP_PARAMSETTING_DEFAULT, TRUE) );

   /* do not catch control-C */
   SCIP_CALL( SCIPsetBoolParam(subscip, "misc/catchctrlc", FALSE) );

#ifndef SCIP_MORE_DEBUG
   /* disable output */
   //SCIP_CALL( SCIPsetIntParam(subscip, "display/verblevel", 0) );
#endif

#ifndef SCIP_DEBUG
   /* disable statistic timing inside sub SCIP */
   SCIP_CALL( SCIPsetBoolParam(subscip, "timing/statistictiming", FALSE) );
#endif

   /* create problem in sub-SCIP */
   /* get name of the original problem and add "comp_nr" */
   (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "%s_comp_%d", SCIPgetProbName(scip), compnr);
   SCIP_CALL( SCIPcreateProb(subscip, name, NULL, NULL, NULL, NULL, NULL, NULL, NULL) );

   /* create variable hashmap */
   SCIP_CALL( SCIPhashmapCreate(&presoldata->varmaps[compnr], SCIPblkmem(subscip), 10 * nvars) );
   varmap = presoldata->varmaps[compnr];

   /* copy variables */
   SCIP_CALL( SCIPcopyVars(scip, subscip, varmap, consmap, FALSE) );

   for( i = 0; i < nconss; ++i )
   {
      /* copy the constraint */
      (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "%s", SCIPconsGetName(conss[i]));
      SCIP_CALL( SCIPgetConsCopy(scip, subscip, conss[i], &newcons, SCIPconsGetHdlr(conss[i]), varmap, consmap, name,
            TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, &success) );

      /* abort if constraint was not successfully copied */
      if( !success )
      {
         SCIP_CALL( SCIPfree(&subscip) );
         presoldata->components[compnr] = NULL;
         presoldata->pluginscopied = FALSE;
         return SCIP_OKAY;
      }

      SCIP_CALL( SCIPaddCons(subscip, newcons) );
      SCIP_CALL( SCIPreleaseCons(subscip, &newcons) );
   }
   /* write the problem, if requested */
   if( presoldata->writeproblems )
   {
      (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "%s_comp_%d.cip", SCIPgetProbName(scip), compnr);
      SCIPdebugMessage("write problem to file %s\n", name);
      SCIP_CALL( SCIPwriteOrigProblem(subscip, name, NULL, FALSE) );

      (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "%s_comp_%d.set", SCIPgetProbName(scip), compnr);
      SCIPdebugMessage("write settings to file %s\n", name);
      SCIP_CALL( SCIPwriteParams(subscip, name, TRUE, TRUE) );
   }

   /* In extended debug mode, we want to be informed if the number of variables was reduced during copying.
    * This might happen, since the components presolver uses SCIPgetConsVars() and then SCIPgetActiveVars() to get the
    * active representation, while SCIPgetConsCopy() might use SCIPgetProbvarLinearSum() and this might cancel out some
    * of the active variables and cannot be avoided. However, we want to notice it and check whether the constraint
    * handler could do something more clever.
    */
#ifdef SCIP_MORE_DEBUG
   if( nvars > SCIPgetNVars(subscip) )
   {
      SCIPinfoMessage(scip, NULL, "copying component %d reduced number of variables: %d -> %d\n", compnr, nvars, SCIPgetNVars(subscip));
   }
#endif

   /* solve the root node of the component */
   SCIP_CALL( solveComponent(scip, presoldata, compnr, 1LL, 0.0, nsolvedprobs, result) );

   return SCIP_OKAY;
}

/** loop over constraints, get active variables and fill directed graph */
static
SCIP_RETCODE fillDigraph(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DIGRAPH*         digraph,            /**< directed graph */
   SCIP_CONS**           conss,              /**< constraints */
   int                   nconss,             /**< number of constraints */
   int*                  firstvaridxpercons, /**< array to store for each constraint the index in the local vars array
                                              *   of the first variable of the constraint */
   SCIP_Bool*            success             /**< flag indicating successful directed graph filling */
   )
{
   SCIP_VAR** consvars;
   int requiredsize;
   int nconsvars;
   int nvars;
   int idx1;
   int idx2;
   int c;
   int v;

   assert(scip != NULL);
   assert(digraph != NULL);
   assert(conss != NULL);
   assert(firstvaridxpercons != NULL);
   assert(success != NULL);

   *success = TRUE;

   nconsvars = 0;
   requiredsize = 0;
   nvars = SCIPgetNVars(scip);

   /* use big buffer for storing active variables per constraint */
   SCIP_CALL( SCIPallocBufferArray(scip, &consvars, nvars) );

   for( c = 0; c < nconss; ++c )
   {
      /* check for reached timelimit */
      if( (c % 1000 == 0) && SCIPisStopped(scip) )
      {
         *success = FALSE;
         break;
      }

      /* get number of variables for this constraint */
      SCIP_CALL( SCIPgetConsNVars(scip, conss[c], &nconsvars, success) );

      if( !(*success) )
         break;

      if( nconsvars > nvars )
      {
         nvars = nconsvars;
         SCIP_CALL( SCIPreallocBufferArray(scip, &consvars, nvars) );
      }

#ifndef NDEBUG
      /* clearing variables array to check for consistency */
      if( nconsvars == nvars )
      {
	 BMSclearMemoryArray(consvars, nconsvars);
      }
      else
      {
	 assert(nconsvars < nvars);
	 BMSclearMemoryArray(consvars, nconsvars + 1);
      }
#endif

      /* get variables for this constraint */
      SCIP_CALL( SCIPgetConsVars(scip, conss[c], consvars, nvars, success) );

      if( !(*success) )
      {
#ifndef NDEBUG
	 /* it looks strange if returning the number of variables was successful but not returning the variables */
	 SCIPwarningMessage(scip, "constraint <%s> returned number of variables but returning variables failed\n", SCIPconsGetName(conss[c]));
#endif
         break;
      }

#ifndef NDEBUG
      /* check if returned variables are consistent with the number of variables that were returned */
      for( v = nconsvars - 1; v >= 0; --v )
	 assert(consvars[v] != NULL);
      if( nconsvars < nvars )
	 assert(consvars[nconsvars] == NULL);
#endif

      /* transform given variables to active variables */
      SCIP_CALL( SCIPgetActiveVars(scip, consvars, &nconsvars, nvars, &requiredsize) );
      assert(requiredsize <= nvars);

      if( nconsvars > 0 )
      {
         idx1 = SCIPvarGetProbindex(consvars[0]);
         assert(idx1 >= 0);

         /* save index of the first variable for later component assignment */
         firstvaridxpercons[c] = idx1;

         if( nconsvars > 1 )
         {
            /* create sparse directed graph
             * sparse means, to add only those edges necessary for component calculation
             */
            for( v = 1; v < nconsvars; ++v )
            {
               idx2 = SCIPvarGetProbindex(consvars[v]);
               assert(idx2 >= 0);

               /* we add only one directed edge, because the other direction is automatically added for component computation */
               SCIP_CALL( SCIPdigraphAddArc(digraph, idx1, idx2, NULL) );
            }
         }
      }
      else
         firstvaridxpercons[c] = -1;
   }

   SCIPfreeBufferArray(scip, &consvars);

   return SCIP_OKAY;
}

#ifdef COMPONENTS_PRINT_STRUCTURE

static
SCIP_RETCODE printStructure(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR**            vars,               /**< variables */
   SCIP_CONS**           conss,              /**< constraints */
   int*                  components,         /**< array with component number for every variable */
   int*                  conscomponents,     /**< array with component number for every constraint */
   int                   nvars,              /**< number of variables */
   int                   nconss,             /**< number of constraints */
   int                   ncomponents         /**< number of components */
   )
{
   char probname[SCIP_MAXSTRLEN];
   char outname[SCIP_MAXSTRLEN];
   char filename[SCIP_MAXSTRLEN];
   char* name;
   FILE* file;
   SCIP_PRESOLDATA* presoldata;
   SCIP_VAR** consvars;
   SCIP_VAR* var;
   int* varidx;
   int* compstartvars;
   int* compstartconss;
   int ncalls;
   int requiredsize;
   int nconsvars;
   int i;
   int c;
   int v;
   SCIP_Bool success;

   SCIP_CALL( SCIPallocBufferArray(scip, &varidx, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &consvars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &compstartvars, ncomponents + 1) );
   SCIP_CALL( SCIPallocBufferArray(scip, &compstartconss, ncomponents + 1) );

   ncalls = SCIPpresolGetNCalls(SCIPfindPresol(scip, PRESOL_NAME));
   compstartvars[0] = 0;
   compstartconss[0] = 0;

   for( v = 0; v < nvars; ++v )
   {
      assert(v == 0 || components[v] == components[v-1] || components[v] == components[v-1]+1);
      varidx[SCIPvarGetProbindex(vars[v])] = v;

      if( v > 0 && components[v] == components[v-1] + 1 )
      {
         compstartvars[components[v]] = v;
      }
   }

   for( v = 0; v < nconss; ++v )
   {
      assert(v == 0 || conscomponents[v] == conscomponents[v-1] || conscomponents[v] == conscomponents[v-1]+1);

      if( v > 0 && conscomponents[v] == conscomponents[v-1] + 1 )
      {
         compstartconss[conscomponents[v]] = v;
      }
   }

   compstartvars[ncomponents] = nvars;
   compstartconss[ncomponents] = nconss;

   /* split problem name */
   (void) SCIPsnprintf(probname, SCIP_MAXSTRLEN, "%s", SCIPgetProbName(scip));
   SCIPsplitFilename(probname, NULL, &name, NULL, NULL);

   /* define file name depending on instance name, number of presolver calls and number of components */
   (void) SCIPsnprintf(outname, SCIP_MAXSTRLEN, "%s_%d_%d", name, ncalls, ncomponents);
   (void) SCIPsnprintf(filename, SCIP_MAXSTRLEN, "%s.gp", outname);

   /* open output file */
   file = fopen(filename, "w");
   if( file == NULL )
   {
      SCIPerrorMessage("cannot create file <%s> for writing\n", filename);
      SCIPprintSysError(filename);
      return SCIP_FILECREATEERROR;
   }

   /* print header of gnuplot file */
   SCIPinfoMessage(scip, file, "set terminal pdf\nset output \"%s.pdf\"\nunset xtics\nunset ytics\n\n", outname);
   SCIPinfoMessage(scip, file, "unset border\nunset key\nset style fill solid 1.0 noborder\nset size ratio -1\n");
   SCIPinfoMessage(scip, file, "set xrange [0:%d]\nset yrange[%d:0]\n", nvars + 1, nconss + 1);

   /* write rectangles around blocks */
   for( i = 0; i < ncomponents; ++i )
   {
      assert(compstartvars[i] <= nvars);
      assert(compstartconss[i] <= nconss);
      assert(compstartvars[i+1] <= nvars);
      assert(compstartconss[i+1] <= nconss);
      SCIPinfoMessage(scip, file, "set object %d rect from %.1f,%.1f to %.1f,%.1f fc rgb \"grey\"\n",
         i + 1, nvars - compstartvars[i] + 0.5, nconss - compstartconss[i] + 0.5,
         nvars - compstartvars[i+1] + 0.5, nconss - compstartconss[i+1] + 0.5);
   }

   /* write the plot header */
   SCIPinfoMessage(scip, file, "plot \"-\" using 1:2:3 with circles fc rgb \"black\"\n");

   /* write data */
   for( c = 0; c < nconss; ++c )
   {
      /* get variables for this constraint */
      SCIP_CALL( SCIPgetConsNVars(scip, conss[c], &nconsvars, &success) );
      SCIP_CALL( SCIPgetConsVars(scip, conss[c], consvars, nvars, &success) );
      assert(success);

      /* transform given variables to active variables */
      SCIP_CALL( SCIPgetActiveVars(scip, consvars, &nconsvars, nvars, &requiredsize) );
      assert(requiredsize <= nvars);

      for( v = 0; v < nconsvars; ++v )
      {
         var = consvars[v];
         SCIPinfoMessage(scip, file, "%d %d 0.5\n", nvars - varidx[SCIPvarGetProbindex(var)], nconss - c);
      }
   }

   /* write file end */
   SCIPinfoMessage(scip, file, "e\n");

   (void) fclose(file);

   SCIPfreeBufferArray(scip, &compstartconss);
   SCIPfreeBufferArray(scip, &compstartvars);
   SCIPfreeBufferArray(scip, &consvars);
   SCIPfreeBufferArray(scip, &varidx);

   return SCIP_OKAY;
}

#endif

/** use components to assign variables and constraints to the subscips
 *  and try to solve all subscips having not too many integer variables
 */
static
SCIP_RETCODE splitProblem(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOLDATA*      presoldata,         /**< presolver data */
   SCIP_CONS**           conss,              /**< constraints */
   SCIP_VAR**            vars,               /**< variables */
   SCIP_Real*            fixvals,            /**< array to store the fixing values */
   int                   nconss,             /**< number of constraints */
   int                   nvars,              /**< number of variables */
   int*                  components,         /**< array with component number for every variable */
   int                   ncomponents,        /**< number of components */
   int*                  firstvaridxpercons, /**< array with index of first variable in vars array for each constraint */
   int*                  nsolvedprobs,       /**< pointer to store the number of solved subproblems */
   int*                  ndeletedvars,       /**< pointer to store the number of deleted variables */
   int*                  ndeletedconss,      /**< pointer to store the number of deleted constraints */
   SCIP_RESULT*          result              /**< pointer to store the result of the presolving call */
   )
{
   SCIP_HASHMAP* consmap;
   SCIP_CONS** compconss;
   SCIP_VAR** compvars;
   SCIP_Real* compfixvals;
   int* conscomponent;
   int ncompconss;
   int nbinvars;
   int nintvars;
   int comp;
   int compvarsstart;
   int compconssstart;
   int nfreevars = 0;
   int v;
   int c;

   assert(scip != NULL);
   assert(presoldata != NULL);
   assert(conss != NULL);
   assert(components != NULL);
   assert(firstvaridxpercons != NULL);
   assert(nsolvedprobs != NULL);
   assert(result != NULL);

   *nsolvedprobs = 0;

   /* hashmap mapping from original constraints to constraints in the sub-SCIPs (for performance reasons) */
   SCIP_CALL( SCIPhashmapCreate(&consmap, SCIPblkmem(scip), 10 * SCIPgetNConss(scip)) );

   SCIP_CALL( SCIPallocClearMemoryArray(scip, &presoldata->components, ncomponents) );
   SCIP_CALL( SCIPallocClearMemoryArray(scip, &presoldata->varmaps, ncomponents) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &presoldata->compvars, ncomponents) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &presoldata->lastdualbound, ncomponents) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &presoldata->lastprimalbound, ncomponents) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &presoldata->lastnodelimit, ncomponents) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &presoldata->laststatus, ncomponents) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &presoldata->lastround, ncomponents) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &presoldata->lastsolindex, ncomponents) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &presoldata->ncompvars, ncomponents) );
   presoldata->ncomponents = 0;
   presoldata->nfeascomps = 0;
   presoldata->nsolvedcomps = 0;

   SCIP_CALL( SCIPcreateSol(scip, &presoldata->bestsol, NULL) );

   SCIP_CALL( SCIPallocBufferArray(scip, &conscomponent, nconss) );

   /* do the mapping from calculated components per variable to corresponding
    * constraints and sort the component-arrays for faster finding the
    * actual variables and constraints belonging to one component
    */
   for( c = 0; c < nconss; c++ )
      conscomponent[c] = (firstvaridxpercons[c] == -1 ? -1 : components[firstvaridxpercons[c]]);

   SCIPsortIntPtr(components, (void**)vars, nvars);
   SCIPsortIntPtr(conscomponent, (void**)conss, nconss);

#ifdef COMPONENTS_PRINT_STRUCTURE
   SCIP_CALL( printStructure(scip, vars, conss, components, conscomponent, nvars, nconss, ncomponents) );
#endif

   compvarsstart = 0;
   compconssstart = 0;

   while( compconssstart < nconss && conscomponent[compconssstart] == -1 )
      ++compconssstart;

   /* loop over all components */
   for( comp = 0; comp < ncomponents && !SCIPisStopped(scip); comp++ )
   {
      nbinvars = 0;
      nintvars = 0;

      /* count variables present in this component and categorize them */
      for( v = compvarsstart; v < nvars && components[v] == comp; ++v )
      {
         /* check whether variable is of binary or integer type */
         if( SCIPvarGetType(vars[v]) == SCIP_VARTYPE_BINARY )
            nbinvars++;
         else if( SCIPvarGetType(vars[v]) == SCIP_VARTYPE_INTEGER )
            nintvars++;
      }

      /* count constraints present in this component */
      c = compconssstart;
      while( c < nconss && conscomponent[c] == comp )
         ++c;

      /* collect some statistical information */
      SCIPstatistic( updateStatisticsComp(presoldata, nbinvars, nintvars) );

      compvars = &(vars[compvarsstart]);
      compfixvals = &(fixvals[compvarsstart]);
      compconss = &(conss[compconssstart]);
      presoldata->ncompvars[comp] = v - compvarsstart;
      ncompconss = c - compconssstart;
      compvarsstart = v;
      compconssstart = c;

      presoldata->lastsolindex[comp] = -1;

      /* the dual fixing presolver will take care of the case ncompconss == 0 */
      assert(ncompconss > 0 || presoldata->ncompvars[comp] == 1);

      SCIP_CALL( SCIPduplicateMemoryArray(scip, &(presoldata->compvars[comp]), compvars, presoldata->ncompvars[comp]) );

      if( ncompconss == 0 )
         nfreevars += ncompvars[comp];

      /* single variable without constraint */
      if( ncompconss == 0 )
      {
         SCIP_Bool infeasible;
         SCIP_Bool fixed;

         /* there is no constraint to connect variables, so there should be only one variable in the component */
         assert(presoldata->ncompvars[comp] == 1);

         ++(presoldata->ncomponents);

         /* is the variable still locked? */
         if( SCIPvarGetNLocksUp(compvars[0]) == 0 && SCIPvarGetNLocksDown(compvars[0]) == 0 )
         {
            /* fix variable to its best bound (or 0.0, if objective is 0) */
            if( SCIPisPositive(scip, SCIPvarGetObj(compvars[0])) )
               compfixvals[0] = SCIPvarGetLbGlobal(compvars[0]);
            else if( SCIPisNegative(scip, SCIPvarGetObj(compvars[0])) )
               compfixvals[0] = SCIPvarGetUbGlobal(compvars[0]);
            else
            {
               compfixvals[0] = 0.0;
               compfixvals[0] = MIN(compfixvals[0], SCIPvarGetUbGlobal(compvars[0])); /*lint !e666*/
               compfixvals[0] = MAX(compfixvals[0], SCIPvarGetLbGlobal(compvars[0])); /*lint !e666*/
            }

#ifdef SCIP_MORE_DEBUG
            SCIPinfoMessage(scip, NULL, "presol components: fix variable <%s>[%g,%g] (locks [%d, %d]) to %g because it occurs in no constraint\n",
               SCIPvarGetName(compvars[0]), SCIPvarGetLbGlobal(compvars[0]), SCIPvarGetUbGlobal(compvars[0]),
               SCIPvarGetNLocksUp(compvars[0]), SCIPvarGetNLocksDown(compvars[0]), compfixvals[0]);
#else
            SCIPdebugMessage("presol components: fix variable <%s>[%g,%g] (locks [%d, %d]) to %g because it occurs in no constraint\n",
               SCIPvarGetName(compvars[0]), SCIPvarGetLbGlobal(compvars[0]), SCIPvarGetUbGlobal(compvars[0]),
               SCIPvarGetNLocksUp(compvars[0]), SCIPvarGetNLocksDown(compvars[0]), compfixvals[0]);
#endif

            SCIP_CALL( SCIPfixVar(scip, compvars[0], compfixvals[0], &infeasible, &fixed) );
            assert(!infeasible);
            assert(fixed);
            ++(*ndeletedvars);
            ++(*nsolvedprobs);

            /* update statistics */
            SCIPstatistic( updateStatisticsSingleVar(presoldata) );
         }
         else
         {
            /* variable is still locked */
            SCIPdebugMessage("strange?! components with single variable variable <%s>[%g,%g] (locks [%d, %d]) and no constraints that won't be fixed due to locks\n",
               SCIPvarGetName(compvars[0]), SCIPvarGetLbGlobal(compvars[0]), SCIPvarGetUbGlobal(compvars[0]),
               SCIPvarGetNLocksUp(compvars[0]), SCIPvarGetNLocksDown(compvars[0]));
         }
      }
      else
      {
         SCIP_RESULT subscipresult;

         assert(ncompconss > 0);

         /* in extended debug mode, we want to be informed about components with single constraints;
          * this is only for noticing this case and possibly handling it within the constraint handler
          */
#ifdef SCIP_MORE_DEBUG
         if( ncompconss == 1 )
         {
            SCIPinfoMessage(scip, NULL, "presol component detected component with a single constraint:\n");
            SCIP_CALL( SCIPprintCons(scip, compconss[0], NULL) );
            SCIPinfoMessage(scip, NULL, ";\n");
         }
#endif

         /* build subscip for one component and try to solve it */
         SCIP_CALL( copyComponent(scip, presoldata, consmap, comp, compconss, ncompconss, compvars,
               compfixvals, presoldata->ncompvars[comp], nbinvars, nintvars, nsolvedprobs, ndeletedvars, ndeletedconss,
               &subscipresult) );

         if( subscipresult == SCIP_CUTOFF || subscipresult == SCIP_UNBOUNDED )
	 {
	    *result = subscipresult;
            break;
	 }

         if( !presoldata->pluginscopied )
            break;
      }
   }

   SCIPfreeBufferArray(scip, &conscomponent);
   SCIPhashmapFree(&consmap);

   return SCIP_OKAY;
}

/** solve the components alternatingly
 */
static
SCIP_RETCODE solveIteratively(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOLDATA*      presoldata,         /**< presolver data */
   int*                  nsolvedprobs,       /**< pointer to store the number of solved subproblems */
   SCIP_RESULT*          result              /**< pointer to store the result of the presolving call */
  )
{
   SCIP_Real lowerbound = -SCIPinfinity(scip);
   SCIP_Real upperbound = SCIPinfinity(scip);
   SCIP_Bool timelimit = FALSE;
   int nfeascomps;
   int ncomponents;
   int comp;
   int c;
   int v;

   ncomponents = presoldata->ncomponents;

   /* iteration */
   for( c = 0; ; ++c )
   {
      nfeascomps = 0;
      upperbound = 0.0;
      lowerbound = 0.0;

      /* count feasible components */
      for( comp = 0; comp < ncomponents; ++comp )
      {
         if( presoldata->components[comp] == NULL )
         {
            assert(presoldata->ncompvars[comp] == 1);
            assert(SCIPvarGetStatus(presoldata->compvars[comp][0]) == SCIP_VARSTATUS_FIXED);

            ++nfeascomps;

            lowerbound += SCIPvarGetObj(presoldata->compvars[comp][0]) * SCIPvarGetLbGlobal(presoldata->compvars[comp][0]);
            upperbound += SCIPvarGetObj(presoldata->compvars[comp][0]) * SCIPvarGetLbGlobal(presoldata->compvars[comp][0]);
         }
         else
         {
            lowerbound += SCIPgetDualbound(presoldata->components[comp]);
            upperbound += SCIPgetPrimalbound(presoldata->components[comp]);

            if( SCIPgetNSols(presoldata->components[comp]) > 0 )
            {
               ++nfeascomps;
            }
         }
      }

      /* construct solution */
      if( nfeascomps == ncomponents )
      {
         SCIP_SOL* sol;
         SCIP_Bool feasible;

         SCIP_CALL( SCIPcreateSol(scip, &sol, NULL) );

         for( comp = 0; comp < ncomponents; ++comp )
         {
            if( presoldata->components[comp] == NULL )
            {
               assert(presoldata->ncompvars[comp] == 1);
               assert(SCIPvarGetStatus(presoldata->compvars[comp][0]) == SCIP_VARSTATUS_FIXED);

               printf("solution of component %d:\n", comp);
               printf("%s     %16.9g\n", SCIPvarGetName(presoldata->compvars[comp][0]), SCIPvarGetLbGlobal(presoldata->compvars[comp][0]));
            }
            else
            {
               SCIP_SOL* bestsol;
               SCIP_VAR* var;
               SCIP_VAR* subvar;

               bestsol = SCIPgetBestSol(presoldata->components[comp]);
               assert(bestsol != NULL);

               // printf("solution of component %d:\n", comp);
               // SCIP_CALL( SCIPprintSol(presoldata->components[comp], bestsol, NULL, FALSE) );

               for( v = 0; v < presoldata->ncompvars[comp]; ++v )
               {
                  var = presoldata->compvars[comp][v];
                  assert(var != NULL);

                  subvar = (SCIP_VAR*)SCIPhashmapGetImage(presoldata->varmaps[comp], (void*)var);
                  assert(subvar != NULL);

                  SCIP_CALL( SCIPsetSolVal(scip, sol, var, SCIPgetSolVal(presoldata->components[comp], bestsol, subvar)) );
               }
            }
         }

         // printf("complete solution:\n");
         // SCIP_CALL( SCIPprintSol(scip, sol, NULL, FALSE) );

         SCIP_CALL( SCIPcheckSolOrig(scip, sol, &feasible, TRUE, TRUE) );

         printf("feasible = %d, obj = %g\n", feasible, SCIPgetSolOrigObj(scip, sol));
         assert(SCIPisFeasEQ(scip, SCIPgetSolOrigObj(scip, sol), SCIPretransformObj(scip, upperbound)));

         SCIP_CALL( SCIPaddSol(scip, sol, &feasible) );

         SCIP_CALL( SCIPfreeSol(scip, &sol) );
      }

      printf("round %d: found feasible solution for %d/%d components\n", c, nfeascomps, ncomponents);
      printf("round %d: %d/%d components solved to optimality\n", c, *nsolvedprobs, ncomponents);

      printf("dualbound: %16.9g; primalbound: %16.9g\n", SCIPretransformObj(scip, lowerbound), nfeascomps == ncomponents ? SCIPretransformObj(scip, upperbound) : SCIPinfinity(scip));

      SCIP_CALL( SCIPupdateLocalLowerbound(scip, lowerbound) );

      if( SCIPisStopped(scip) || timelimit || SCIPisEQ(scip, lowerbound, upperbound) )
         break;

      timelimit = TRUE;

      /* solve sub-SCIPs alternatingly */
      for( comp = 0; comp < ncomponents && !SCIPisStopped(scip); ++comp )
      {
         SCIP_Longint nodelimit;
         SCIP_Real gaplimit;
         SCIP_RESULT subscipresult;

         /* component was already solved */
         if( presoldata->laststatus[comp] == SCIP_STATUS_OPTIMAL )
         {
            assert(presoldata->components[comp] == NULL);
            continue;
         }

         /* first, we want to reach feasibility for all components */
         if( presoldata->nfeascomps < ncomponents && SCIPgetNSols(presoldata->components[comp]) > 0 )
            continue;

         /* if the gap is too small, skip this component (except for every fifth round) */
         if( c % 5 != 0 && SCIPgetNSols(presoldata->components[comp]) > 0 &&
            SCIPgetPrimalbound(presoldata->components[comp]) - SCIPgetDualbound(presoldata->components[comp])
            < (SCIPgetPrimalbound(scip) - SCIPgetDualbound(scip)) / (2.0 * (ncomponents - *nsolvedprobs)) )
            continue;

         nodelimit = 2 * SCIPgetNNodes(presoldata->components[comp]);
         nodelimit = MAX(nodelimit, 500LL);

         /* set a gap limit of half the current gap (at most 10%) */
         if( SCIPgetGap(presoldata->components[comp]) < 0.2 )
            gaplimit = 0.5 * SCIPgetGap(presoldata->components[comp]);
         else
            gaplimit = 0.1;

         /* continue solving the component */
         SCIP_CALL( solveComponent(scip, presoldata, comp, nodelimit, gaplimit, nsolvedprobs, &subscipresult) );

         if( subscipresult == SCIP_CUTOFF || subscipresult == SCIP_UNBOUNDED )
	 {
	    *result = subscipresult;
            break;
	 }

         timelimit = timelimit && (presoldata->laststatus[comp] == SCIP_STATUS_TIMELIMIT);
      }

      if( *result == SCIP_CUTOFF || *result == SCIP_UNBOUNDED )
      {
         break;
      }
   }

   return SCIP_OKAY;
}

/** performs presolving by searching for components */
static
SCIP_RETCODE presolComponents(
   SCIP*                 scip,               /**< SCIP main data structure */
   SCIP_PRESOL*          presol,             /**< the presolver itself */
   int*                  nfixedvars,         /**< counter to increase by the number of fixed variables */
   int*                  ndelconss,          /**< counter to increase by the number of deleted constrains */
   SCIP_RESULT*          result              /**< pointer to store the result of the presolving call */
   )
{
   SCIP_PRESOLDATA* presoldata;
   SCIP_CONS** conss;
   SCIP_CONS** tmpconss;
   SCIP_Bool success;
   int nconss;
   int ntmpconss;
   int nvars;
   int ncomponents;
   int ndeletedconss;
   int ndeletedvars;
   int nsolvedprobs;
   int c;

   assert(scip != NULL);
   assert(presol != NULL);
   assert(result != NULL);

   *result = SCIP_DIDNOTRUN;

   if( SCIPgetStage(scip) != SCIP_STAGE_PRESOLVING || SCIPinProbing(scip) )
      return SCIP_OKAY;

   /* do not run, if not all variables are explicitly known */
   if( SCIPgetNActivePricers(scip) > 0 )
      return SCIP_OKAY;

   presoldata = SCIPpresolGetData(presol);
   assert(presoldata != NULL);

   nvars = SCIPgetNVars(scip);

   /* we do not want to run, if there are no variables left */
   if( nvars == 0 )
      return SCIP_OKAY;

   /* the presolver should be executed only if it didn't run so far or the number of variables was significantly reduced
    * since tha last run
    */
   if( (presoldata->didsearch && nvars > (1 - presoldata->reldecrease) * presoldata->lastnvars) )
      return SCIP_OKAY;

   /* if we were not able to copy all plugins, we cannot run the components presolver */
   if( !presoldata->pluginscopied )
      return SCIP_OKAY;

   /* only call the component presolver, if presolving would be stopped otherwise */
   if( !SCIPisPresolveFinished(scip) )
      return SCIP_OKAY;

   /* check for a reached timelimit */
   if( SCIPisStopped(scip) )
      return SCIP_OKAY;

   SCIPstatistic( resetStatistics(scip, presoldata) );

   *result = SCIP_DIDNOTFIND;
   presoldata->didsearch = TRUE;

   ncomponents = 0;
   ndeletedvars = 0;
   ndeletedconss = 0;
   nsolvedprobs = 0;

   /* collect checked constraints for component presolving */
   ntmpconss = SCIPgetNConss(scip);
   tmpconss = SCIPgetConss(scip);
   SCIP_CALL( SCIPallocBufferArray(scip, &conss, ntmpconss) );
   nconss = 0;
   for( c = 0; c < ntmpconss; c++ )
   {
      if( SCIPconsIsChecked(tmpconss[c]) )
      {
         conss[nconss] = tmpconss[c];
         nconss++;
      }
   }

   if( nvars > 1 && nconss > 1 )
   {
      SCIP_DIGRAPH* digraph;
      SCIP_VAR** vars;
      SCIP_Real* fixvals;
      int* firstvaridxpercons;
      int* varlocks;
      int v;

      /* copy variables into a local array */
      SCIP_CALL( SCIPallocBufferArray(scip, &vars, nvars) );
      SCIP_CALL( SCIPallocBufferArray(scip, &fixvals, nvars) );
      SCIP_CALL( SCIPallocBufferArray(scip, &firstvaridxpercons, nconss) );
      SCIP_CALL( SCIPallocBufferArray(scip, &varlocks, nvars) );
      BMScopyMemoryArray(vars, SCIPgetVars(scip), nvars);

      /* count number of varlocks for each variable (up + down locks) and multiply it by 2;
       * that value is used as an estimate of the number of arcs incident to the variable's node in the digraph
       * to be safe, we double this value
       */
      for( v = 0; v < nvars; ++v )
         varlocks[v] = 4 * (SCIPvarGetNLocksDown(vars[v]) + SCIPvarGetNLocksUp(vars[v]));

      /* create and fill directed graph */
      SCIP_CALL( SCIPdigraphCreate(&digraph, nvars) );
      SCIP_CALL( SCIPdigraphSetSizes(digraph, varlocks) );
      SCIP_CALL( fillDigraph(scip, digraph, conss, nconss, firstvaridxpercons, &success) );

      if( success )
      {
         int* components;

         SCIP_CALL( SCIPallocBufferArray(scip, &components, nvars) );

         /* compute independent components */
         SCIP_CALL( SCIPdigraphComputeUndirectedComponents(digraph, 1, components, &ncomponents) );

#ifndef NDEBUG
         SCIPverbMessage(scip, SCIP_VERBLEVEL_FULL, NULL, "presol components found %d undirected components\n", ncomponents);
#else
         SCIPdebugMessage("presol components found %d undirected components\n", ncomponents);
#endif

         /* We want to sort the components in increasing size (number of variables).
          * Therefore, we now get the number of variables for each component, and rename the components
          * such that for i < j, component i has no more variables than component j.
          * @todo Perhaps sort the components by the number of binary/integer variables?
          */
         if( ncomponents > 1 )
         {
            int* compsize;
            int* permu;
            int* compvars;
            int ncompvars;
            int nbinvars;
            int nintvars;
            int ncontvars;

            SCIP_CALL( SCIPallocBufferArray(scip, &compsize, ncomponents) );
            SCIP_CALL( SCIPallocBufferArray(scip, &permu, ncomponents) );

            /* get number of variables in the components */
            for( c = 0; c < ncomponents; ++c )
            {
               SCIPdigraphGetComponent(digraph, c, &compvars, &ncompvars);
               permu[c] = c;
               nbinvars = 0;
               nintvars = 0;

               for( v = 0; v < ncompvars; ++v )
               {
                  /* check whether variable is of binary or integer type */
                  if( SCIPvarGetType(vars[compvars[v]]) == SCIP_VARTYPE_BINARY )
                     nbinvars++;
                  else if( SCIPvarGetType(vars[compvars[v]]) == SCIP_VARTYPE_INTEGER )
                     nintvars++;
               }
               ncontvars = ncompvars - nintvars - nbinvars;
               compsize[c] = (int)((1000 * (nbinvars + presoldata->intfactor * nintvars) + (950.0 * ncontvars)/nvars));
            }

            /* get permutation of component numbers such that the size of the components is increasing */
            SCIPsortIntInt(compsize, permu, ncomponents);

            /* now, we need the reverse direction, i.e., for each component number, we store its new number
             * such that the components are sorted; for this, we abuse the compsize array
             */
            for( c = 0; c < ncomponents; ++c )
               compsize[permu[c]] = c;

            /* for each variable, replace the old component number by the new one */
            for( c = 0; c < nvars; ++c )
               components[c] = compsize[components[c]];

            SCIPfreeBufferArray(scip, &permu);
            SCIPfreeBufferArray(scip, &compsize);

            /* create subproblems from independent components and solve them in dependence of their size */
            SCIP_CALL( splitProblem(scip, presoldata, conss, vars, fixvals, nconss, nvars, components, ncomponents, firstvaridxpercons,
                  &nsolvedprobs, &ndeletedvars, &ndeletedconss, result) );

            (*nfixedvars) += ndeletedvars;
            (*ndelconss) += ndeletedconss;

            if( presoldata->pluginscopied )
            {
               solveIteratively(scip, presoldata, &nsolvedprobs, result);
            }

            /* free components */
            for( c = 0; c < ncomponents; ++c )
            {
               if( presoldata->components[c] != NULL )
               {
                  SCIP_CALL( SCIPfree(&presoldata->components[c]) );
               }
            }

         }

         SCIPfreeBufferArray(scip, &components);
      }

      SCIPdigraphFree(&digraph);

      SCIPfreeBufferArray(scip, &varlocks);
      SCIPfreeBufferArray(scip, &firstvaridxpercons);
      SCIPfreeBufferArray(scip, &fixvals);
      SCIPfreeBufferArray(scip, &vars);
   }

   SCIPfreeBufferArray(scip, &conss);

   /* update result to SCIP_SUCCESS, if reductions were found;
    * if result was set to infeasible or unbounded before, leave it as it is
    */
   if( (ndeletedvars > 0 || ndeletedconss > 0) && ((*result) == SCIP_DIDNOTFIND) )
      *result = SCIP_SUCCESS;

   /* print statistics */
   SCIPstatistic( printStatistics(presoldata) );

   SCIPdebugMessage("%d components, %d solved, %d deleted constraints, %d deleted variables, result = %d\n",
      ncomponents, nsolvedprobs, ndeletedconss, ndeletedvars, *result);
#ifdef NDEBUG
   SCIPstatisticMessage("%d components, %d solved, %d deleted constraints, %d deleted variables, result = %d\n",
      ncomponents, nsolvedprobs, ndeletedconss, ndeletedvars, *result);
#endif

   presoldata->lastnvars = SCIPgetNVars(scip);

   return SCIP_OKAY;
}


/*
 * Callback methods of presolver
 */

/** destructor of presolver to free user data (called when SCIP is exiting) */
static
SCIP_DECL_PRESOLFREE(presolFreeComponents)
{  /*lint --e{715}*/
   SCIP_PRESOLDATA* presoldata;

   /* free presolver data */
   presoldata = SCIPpresolGetData(presol);
   assert(presoldata != NULL);

   SCIPfreeMemory(scip, &presoldata);
   SCIPpresolSetData(presol, NULL);

   return SCIP_OKAY;
}

/** initialization method of presolver (called after problem was transformed) */
static
SCIP_DECL_PRESOLINIT(presolInitComponents)
{  /*lint --e{715}*/
   SCIP_PRESOLDATA* presoldata;

   /* free presolver data */
   presoldata = SCIPpresolGetData(presol);
   assert(presoldata != NULL);

   /* initialize statistics */
   SCIPstatistic( SCIP_CALL( initStatistics(scip, presoldata) ) );

   presoldata->didsearch = FALSE;
   presoldata->pluginscopied = TRUE;

   return SCIP_OKAY;
}

#ifdef SCIP_STATISTIC
/** deinitialization method of presolver (called before transformed problem is freed) */
static
SCIP_DECL_PRESOLEXIT(presolExitComponents)
{  /*lint --e{715}*/
   SCIP_PRESOLDATA* presoldata;

   /* free presolver data */
   presoldata = SCIPpresolGetData(presol);
   assert(presoldata != NULL);

   SCIPstatistic( freeStatistics(scip, presoldata) );

   return SCIP_OKAY;
}
#endif


/** execution method of presolver */
static
SCIP_DECL_PRESOLEXEC(presolExecComponents)
{  /*lint --e{715}*/

   SCIP_CALL( presolComponents(scip, presol, nfixedvars, ndelconss, result) );

   return SCIP_OKAY;
}


/*
 * presolver specific interface methods
 */

/** creates the components presolver and includes it in SCIP */
SCIP_RETCODE SCIPincludePresolComponents(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_PRESOLDATA* presoldata;
   SCIP_PRESOL* presol;

   /* create components presolver data */
   SCIP_CALL( SCIPallocMemory(scip, &presoldata) );

   /* include presolver */
   SCIP_CALL( SCIPincludePresolBasic(scip, &presol, PRESOL_NAME, PRESOL_DESC, PRESOL_PRIORITY, PRESOL_MAXROUNDS,
         PRESOL_TIMING, presolExecComponents, presoldata) );

   /* currently, the components presolver is not copied; if a copy callback is added, we need to avoid recursion
    * by one of the following means:
    * - turn off the components presolver in SCIPsetSubscipsOff()
    * - call SCIPsetSubscipsOff() for the component sub-SCIP
    * - disable the components presolver in the components sub-SCIP
    */
   SCIP_CALL( SCIPsetPresolCopy(scip, presol, NULL) );
   SCIP_CALL( SCIPsetPresolFree(scip, presol, presolFreeComponents) );
   SCIP_CALL( SCIPsetPresolInit(scip, presol, presolInitComponents) );
   SCIPstatistic( SCIP_CALL( SCIPsetPresolExit(scip, presol, presolExitComponents) ) );

   /* add presolver parameters */
   SCIP_CALL( SCIPaddBoolParam(scip,
         "presolving/components/writeproblems",
         "should the single components be written as an .cip-file?",
         &presoldata->writeproblems, FALSE, DEFAULT_WRITEPROBLEMS, NULL, NULL) );
   SCIP_CALL( SCIPaddIntParam(scip,
         "presolving/components/maxintvars",
         "maximum number of integer (or binary) variables to solve a subproblem directly (-1: unlimited)",
         &presoldata->maxintvars, FALSE, DEFAULT_MAXINTVARS, -1, INT_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddLongintParam(scip,
         "presolving/components/nodelimit",
         "maximum number of nodes to be solved in subproblems",
         &presoldata->nodelimit, FALSE, DEFAULT_NODELIMIT, -1LL, SCIP_LONGINT_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip,
         "presolving/components/intfactor",
         "the weight of an integer variable compared to binary variables",
         &presoldata->intfactor, FALSE, DEFAULT_INTFACTOR, 0.0, SCIP_REAL_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip,
         "presolving/components/reldecrease",
         "percentage by which the number of variables has to be decreased after the last component solving to allow running again (1.0: do not run again)",
         &presoldata->reldecrease, FALSE, DEFAULT_RELDECREASE, 0.0, 1.0, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip,
         "presolving/components/feastolfactor",
         "factor to increase the feasibility tolerance of the main SCIP in all sub-SCIPs, default value 1.0",
         &presoldata->feastolfactor, TRUE, DEFAULT_FEASTOLFACTOR, 0.0, 1000000.0, NULL, NULL) );

   return SCIP_OKAY;
}
