/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2011 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#define SCIP_DEBUG

/**@file   presol_components.c
 * @brief  solve independent components in advance
 * @author Dieter Weninger
 * @author Gerald Gamrath
 *
 * TODO: simulation of presolving without solve
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>

#include "scip/presol_components.h"


#define PRESOL_NAME            "components"
#define PRESOL_DESC            "components presolver"
#define PRESOL_PRIORITY        -9200000      /**< priority of the presolver (>= 0: before, < 0: after constraint handlers); combined with propagators */
#define PRESOL_MAXROUNDS              0      /**< maximal number of presolving rounds the presolver participates in (-1: no limit) */
#define PRESOL_DELAY               TRUE      /**< should presolver be delayed, if other presolvers found reductions? */

#define DEFAULT_SEARCH             TRUE      /**< should be searched for components? */
#define DEFAULT_WRITEPROBLEMS     FALSE      /**< should the single components be written as an .lp-file? */
#define DEFAULT_MAXINTVARS           20      /**< maximum number of integer (or binary) variables to solve a subproblem directly (-1: no solving) */
#define DEFAULT_NODELIMIT         10000      /**< maximum number of nodes to be solved in subproblems */
#define DEFAULT_INTFACTOR             1      /**< the weight of an integer variable compared to binary variables */


/*
 * Data structures
 */

/** control parameters */
struct SCIP_PresolData
{
   SCIP_Bool             dosearch;           /** should be searched for components? */
   SCIP_Bool             didsearch;          /** did the presolver already search for components? */
   SCIP_Bool             writeproblems;      /** should the single components be written as an .lp-file? */
   int                   maxintvars;         /** maximum number of integer (or binary) variables to solve a subproblem directly (-1: no solving) */
   SCIP_Longint          nodelimit;          /** maximum number of nodes to be solved in subproblems */
   SCIP_Real             intfactor;          /** the weight of an integer variable compared to binary variables */
};

/*
 * Local methods
 */

/** copies a connected component consisting of the given constraints and variables into a sub-SCIP
 *  and tries to solve the sub-SCIP to optimality
 */
static
SCIP_RETCODE solveComponent(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOLDATA*      presoldata,         /**< presolver data */
   SCIP_HASHMAP*         consmap,            /**< constraint hashmap used to improve performance */
   int                   compnr,             /**< number of the component */
   SCIP_CONS**           conss,              /**< constraints contained in this component */
   int                   nconss,             /**< number of constraints contained in this component */
   SCIP_VAR**            vars,               /**< variables contained in this component */
   int                   nvars,              /**< number of variables contained in this component */
   int*                  nsolvedprobs,       /**< pointer to increase, if the subproblem was solved */
   SCIP_Real*            subsolvetime,       /**< pointer to store time needed to solve the subproblem */
   int*                  nconstodelete,      /**< number of constraints for deletion */
   SCIP_CONS**           constodelete,       /**< constraints for deletion */
   int*                  nvarstofix,         /**< number of variables for fixing */
   SCIP_VAR**            varstofix,          /**< variables for fixing */
   SCIP_Real*            varsfixvalues,      /**< fixing values of the variables */
   SCIP_RESULT*          result              /**< pointer to store the result of the presolving call */
   )
{
   char name[SCIP_MAXSTRLEN];
   SCIP* subscip;
   SCIP_HASHMAP* varmap;
   SCIP_CONS* newcons;
   SCIP_Real timelimit;
   SCIP_Real memorylimit;
   SCIP_Bool success;
   int i;

   assert(scip != NULL);
   assert(presoldata != NULL);
   assert(conss != NULL);
   assert(nconss > 0);
   assert(vars != NULL);
   assert(nvars > 0);
   assert(nsolvedprobs != NULL);
   assert(subsolvetime != NULL);
   assert(nconstodelete != NULL);
   assert(constodelete != NULL);
   assert(nvarstofix != NULL);
   assert(varstofix != NULL);
   assert(varsfixvalues != NULL);

   /* check whether there is enough time and memory left */
   SCIP_CALL( SCIPgetRealParam(scip, "limits/time", &timelimit) );
   if( !SCIPisInfinity(scip, timelimit) )
      timelimit -= SCIPgetSolvingTime(scip);
   SCIP_CALL( SCIPgetRealParam(scip, "limits/memory", &memorylimit) );
   if( !SCIPisInfinity(scip, memorylimit) )
      memorylimit -= SCIPgetMemUsed(scip)/1048576.0;
   if( timelimit <= 0.0 || memorylimit <= 0.0 )
      goto TERMINATE;

   /* create sub-SCIP */
   SCIP_CALL( SCIPcreate(&subscip) );

   /* create variable hashmap */
   SCIP_CALL( SCIPhashmapCreate(&varmap, SCIPblkmem(scip), 10 * nvars) );

   /* copy plugins, we omit pricers (because we do not run if there are active pricers) and dialogs */
   success = TRUE;
   SCIP_CALL( SCIPcopyPlugins(scip, subscip, TRUE, FALSE, TRUE, TRUE, TRUE, TRUE, TRUE,
         TRUE, TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, TRUE, &success) );

   if( !success )
      goto TERMINATE;

   /* copy parameter settings */
   SCIP_CALL( SCIPcopyParamSettings(scip, subscip) );

   /* set time and memory limit for the subproblem */
   SCIP_CALL( SCIPsetRealParam(subscip, "limits/time", timelimit) );
   SCIP_CALL( SCIPsetRealParam(subscip, "limits/memory", memorylimit) );

   /* set node limit */
   SCIP_CALL( SCIPsetLongintParam(subscip, "limits/nodes", presoldata->nodelimit) );

   /* set gap limit to 0 */
   SCIP_CALL( SCIPsetRealParam(subscip, "limits/gap", 0.0) );

   /* reduce the effort spent for hash tables */
   SCIP_CALL( SCIPsetBoolParam(subscip, "misc/usevartable", FALSE) );
   SCIP_CALL( SCIPsetBoolParam(subscip, "misc/useconstable", FALSE) );
   SCIP_CALL( SCIPsetBoolParam(subscip, "misc/usesmalltables", TRUE) );

   /* do not catch control-C */
   SCIP_CALL( SCIPsetBoolParam(subscip, "misc/catchctrlc", FALSE) );

   /* disable output */
   SCIP_CALL( SCIPsetIntParam(subscip, "display/verblevel", SCIP_VERBLEVEL_NONE) );

   /* create problem in sub-SCIP */
   /* get name of the original problem and add "comp_nr" */
   (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "%s_comp_%d", SCIPgetProbName(scip), compnr);
   SCIP_CALL( SCIPcreateProb(subscip, name, NULL, NULL, NULL, NULL, NULL, NULL, NULL) );

   for( i = 0; i < nconss; ++i )
   {
      /* copy the constraint */
      (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "%s", SCIPconsGetName(conss[i]));
      SCIP_CALL( SCIPgetConsCopy(scip, subscip, conss[i], &newcons, SCIPconsGetHdlr(conss[i]), varmap, consmap, name,
            TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, &success) );

      /* break if constraint was not successfully copied */
      if( !success )
         goto TERMINATE;

      SCIP_CALL( SCIPaddCons(subscip, newcons) );
      SCIP_CALL( SCIPreleaseCons(subscip, &newcons) );
   }

   /* write the problem, if requested */
   if( presoldata->writeproblems )
   {
      (void) SCIPsnprintf(name, SCIP_MAXSTRLEN, "%s_comp_%d.lp", SCIPgetProbName(scip), compnr);
      SCIPdebugMessage("write problem to file %s\n", name);
      SCIP_CALL( SCIPwriteOrigProblem(subscip, name, NULL, FALSE) );
   }

   /* solve the subproblem */
   SCIP_CALL( SCIPsolve(subscip) );

   *subsolvetime += SCIPgetSolvingTime(subscip);

   if( SCIPgetStatus(subscip) == SCIP_STATUS_OPTIMAL )
   {
      SCIP_SOL* sol;

      ++(*nsolvedprobs);

      sol = SCIPgetBestSol(subscip);

      /* memorize variables for later fixing */
      for( i = 0; i < nvars; ++i )
      {
         assert( SCIPhashmapExists(varmap, vars[i]) );
         varstofix[*nvarstofix] = vars[i];
         varsfixvalues[*nvarstofix] = SCIPgetSolVal(subscip, sol, SCIPhashmapGetImage(varmap, vars[i]));
         (*nvarstofix)++;
      }

      /* memorize constraints for later deletion */
      for( i = 0; i < nconss; ++i )
      {
         constodelete[*nconstodelete] = conss[i];
         (*nconstodelete)++;
      }
   }
   else if( SCIPgetStatus(subscip) == SCIP_STATUS_INFEASIBLE )
   {
      *result = SCIP_CUTOFF;
   }
   else if( SCIPgetStatus(subscip) == SCIP_STATUS_UNBOUNDED )
   {
      /* TODO: store unbounded ray in original SCIP data structure */
      *result = SCIP_UNBOUNDED;
   }
   else
   {
      SCIPdebugMessage("++++++++++++++ sub-SCIP for component %d not solved (status=%d, time=%.2f): %d vars (%d bin, %d int, %d impl, %d cont), %d conss\n",
         compnr, SCIPgetStatus(subscip), SCIPgetSolvingTime(subscip), nvars, SCIPgetNBinVars(subscip), SCIPgetNIntVars(subscip), SCIPgetNImplVars(subscip),
         SCIPgetNContVars(subscip), nconss);
   }

 TERMINATE:
   SCIP_CALL( SCIPfree(&subscip) );
   SCIPhashmapFree(&varmap);

   return SCIP_OKAY;
}

/** loop over constraints, get active variables and fill directed graph */
static
SCIP_RETCODE fillDigraph(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_DIGRAPH*         digraph,            /**< directed graph */
   SCIP_CONS**           conss,              /**< constraints */
   int                   nconss,             /**< number of constraints */
   int*                  firstvaridxpercons, /**< problem index of first variable per constraint */
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
      /* get number of variables for this cconstraint */
      SCIP_CALL( SCIPgetConsNVars(scip, conss[c], &nconsvars, success) );

      if( !(*success) )
         break;

      if( nconsvars > nvars )
      {
         nvars = nconsvars;
         SCIP_CALL( SCIPreallocBufferArray(scip, &consvars, nvars) );
      }

      /* get variables for this constraint */
      SCIP_CALL( SCIPgetConsVars(scip, conss[c], consvars, nvars, success) );

      if( !(*success) )
         break;

      /* transform given variables to active variables */
      SCIP_CALL( SCIPgetActiveVars(scip, consvars, &nconsvars, nvars, &requiredsize) );
      assert(requiredsize <= nvars);

      idx1 = SCIPvarGetProbindex(consvars[0]);
      assert(idx1 >= 0);

      /* save problem index of the first variable for later component assignment */
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

            /* we add a directed edge in both directions */
            SCIP_CALL( SCIPdigraphAddEdge(digraph, idx1, idx2) );
            SCIP_CALL( SCIPdigraphAddEdge(digraph, idx2, idx1) );
         }
      }
   }

   SCIPfreeBufferArray(scip, &consvars);

   return SCIP_OKAY;
}

/** calculate frequency distribution of component sizes
 *  in dependence of the number of discrete variables
 */
static
void updateStatistics(
   int                   nbinvars,           /**< number of binary variables */
   int                   nintvars,           /**< number of integer variables */
   int*                  statistics          /**< array saving statistical information */
   )
{
   int ndiscretevars;

   assert(statistics != NULL);

   ndiscretevars = nbinvars + nintvars;

   if( 0 <= ndiscretevars && ndiscretevars < 21 )
   {
      statistics[0]++;
   }
   else if( 20 < ndiscretevars && ndiscretevars < 51 )
   {
      statistics[1]++;
   }
   else if( 50 < ndiscretevars && ndiscretevars < 101 )
   {
      statistics[2]++;
   }
   else if( 100 < ndiscretevars )
   {
      statistics[3]++;
   }
}

/** use components to assign variables and constraints to the subscips
 *  and try to solve all subscips having not too many integer variables
 */
static
SCIP_RETCODE splitProblem(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOLDATA*      presoldata,         /**< presolver data */
   SCIP_CONS**           conss,              /**< constraints */
   int                   nconss,             /**< number of constraints */
   int*                  components,         /**< array with component number for every variable */
   int*                  ncomponents,        /**< size of array with component numbers */
   int*                  firstvaridxpercons, /**< problem index of first variable per constraint */
   int*                  nsolvedprobs,       /**< number of solved subproblems */
   int*                  nconstodelete,      /**< number of constraints for deletion */
   SCIP_CONS**           constodelete,       /**< constraints for deletion */
   int*                  nvarstofix,         /**< number of variables for fixing */
   SCIP_VAR**            varstofix,          /**< variables for fixing */
   SCIP_Real*            varsfixvalues,      /**< variables fixing values */
   int*                  statistics,         /**< array holding some statistical information */
   SCIP_RESULT*          result              /**< pointer to store the result of the presolving call */
  )
{
   SCIP_HASHMAP* consmap; /** hashmap mapping from original constraints to constraints in the sub-SCIPs (for performance reasons) */
   SCIP_VAR** vars;
   SCIP_CONS** tmpconss;
   SCIP_VAR** tmpvars;
   int* conscomponent;
   int* considx;
   int* varscomponent;
   int* varsidx;
   SCIP_Real subsolvetime;
   int nvars;
   int ntmpconss;
   int ntmpvars;
   int nbinvars;
   int nintvars;
   int comp;
   int v;
   int c;

   assert(scip != NULL);
   assert(presoldata != NULL);
   assert(conss != NULL);
   assert(components != NULL);
   assert(ncomponents != NULL);
   assert(firstvaridxpercons != NULL);
   assert(nsolvedprobs != NULL);
   assert(nconstodelete != NULL);
   assert(constodelete != NULL);
   assert(nvarstofix != NULL);
   assert(varstofix != NULL);
   assert(varsfixvalues != NULL);
   assert(statistics != NULL);
   assert(result != NULL);

   *nsolvedprobs = 0;
   subsolvetime = 0.0;

   nvars = SCIPgetNVars(scip);
   vars = SCIPgetVars(scip);

   SCIP_CALL( SCIPhashmapCreate(&consmap, SCIPblkmem(scip), 10 * SCIPgetNConss(scip)) );

   SCIP_CALL( SCIPallocBufferArray(scip, &tmpconss, nconss) );
   SCIP_CALL( SCIPallocBufferArray(scip, &tmpvars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &conscomponent, nconss) );
   SCIP_CALL( SCIPallocBufferArray(scip, &considx, nconss) );
   SCIP_CALL( SCIPallocBufferArray(scip, &varscomponent, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &varsidx, nvars) );

   /* do the mapping from calculated components per variable to corresponding
    * constraints and sort the component-arrays for faster finding the
    * actual variables and constraints belonging to one component
    */
   for( c = 0; c < nconss; c++ )
   {
      conscomponent[c] = components[firstvaridxpercons[c]];
      considx[c] = c;
   }
   for( v = 0; v < nvars; v++ )
   {
      varscomponent[v] = components[v];
      varsidx[v] = v;
   }
   SCIPsortIntInt(varscomponent, varsidx, nvars);
   SCIPsortIntInt(conscomponent, considx, nconss);

   v = 0;
   c = 0;

   /* loop over all components
    * start loop from 1 because components are numbered form 1..n
    */
   for( comp = 1; comp <= *ncomponents; comp++ )
   {
      ntmpconss = 0;
      ntmpvars = 0;
      nbinvars = 0;
      nintvars = 0;

      /* get variables present in this component */
      while( v < nvars && varscomponent[v] == comp )
      {
         /* variable is present in this component */
         tmpvars[ntmpvars] = vars[varsidx[v]];

         /* check whether variable is of binary or integer type */
         if( SCIPvarGetType(tmpvars[ntmpvars]) == SCIP_VARTYPE_BINARY )
            nbinvars++;
         else if( SCIPvarGetType(tmpvars[ntmpvars]) == SCIP_VARTYPE_INTEGER )
            nintvars++;

         ++ntmpvars;
         ++v;
      }
      assert(ntmpvars != 0);

      /* get constraints present in this component */
      while( c < nconss && conscomponent[c] == comp )
      {
         /* constraint is present in this component */
         tmpconss[ntmpconss] = conss[considx[c]];
         ++ntmpconss;
         ++c;
      }

      /* collect some statistical information */
      updateStatistics(nbinvars, nintvars, statistics);

      if( (nbinvars + presoldata->intfactor * nintvars <= presoldata->maxintvars) || presoldata->writeproblems )
      {
         /* single variable without constraint */
         if( ntmpconss == 0 )
         {
            /* there is no constraint to connect variables, so there should be only one */
            assert(ntmpvars == 1);

            /* fix variable to its best bound */
            varstofix[*nvarstofix] = tmpvars[0];
            if( SCIPisPositive(scip, SCIPvarGetObj(tmpvars[0])) )
               varsfixvalues[*nvarstofix] = SCIPvarGetLbGlobal(tmpvars[0]);
            else if( SCIPisNegative(scip, SCIPvarGetObj(tmpvars[0])) )
               varsfixvalues[*nvarstofix] = SCIPvarGetUbGlobal(tmpvars[0]);
            else
               varsfixvalues[*nvarstofix] = 0.0;
            (*nvarstofix)++;
         }
         /* single constraint without variables */
         else if( ntmpvars == 0 )
         {
            /* there is no variable connecting constraints, so there should be only one */
            assert(ntmpconss == 1);

            constodelete[*nconstodelete] = tmpconss[0];
            (*nconstodelete)++;
         }
         else
         {
            assert(ntmpconss > 0);
            assert(ntmpvars > 0 );

            /* build subscip for one component and try to solve it */
            SCIP_CALL( solveComponent(scip, presoldata, consmap, comp, tmpconss, ntmpconss,
                  tmpvars, ntmpvars, nsolvedprobs, &subsolvetime, nconstodelete, constodelete,
                  nvarstofix, varstofix, varsfixvalues, result) );

            if( *result == SCIP_CUTOFF )
               break;
         }
      }
      else
      {
         SCIPdebugMessage("++++++++++++++ sub-SCIP for component %d not created: %d vars (%d bin, %d int, %d cont), %d conss\n",
            comp, ntmpvars, nbinvars, nintvars, ntmpvars - nintvars - nbinvars, ntmpconss);
      }
   }

   SCIPfreeBufferArray(scip, &varsidx);
   SCIPfreeBufferArray(scip, &varscomponent);
   SCIPfreeBufferArray(scip, &considx);
   SCIPfreeBufferArray(scip, &conscomponent);
   SCIPfreeBufferArray(scip, &tmpvars);
   SCIPfreeBufferArray(scip, &tmpconss);
   SCIPhashmapFree(&consmap);

   return SCIP_OKAY;
}

/** do variable fixing and constraint deletion at the end,
 *  since a variable fixing can change the current and the
 *  subsequent slots in the vars array
 */
static
SCIP_RETCODE fixVarsDeleteConss(
   SCIP*                 scip,               /**< SCIP data structure */
   int                   nconsstodelete,     /**< number of constraints for deletion */
   SCIP_CONS**           consstodelete,      /**< constraints for deletion */
   int                   nvarstofix,         /**< number of variables for fixing */
   SCIP_VAR**            varstofix,          /**< variables for fixing */
   SCIP_Real*            varsfixvalues,      /**< fixing values of the variables */
   int*                  ndeletedcons,       /**< number of deleted constraints by component presolver */
   int*                  ndeletedvars        /**< number of fixed variables by component presolver */
   )
{
   int i;
   SCIP_Bool infeasible;
   SCIP_Bool fixed;

   assert(scip != NULL);
   assert(consstodelete != NULL);
   assert(varstofix != NULL);
   assert(varsfixvalues != NULL);
   assert(ndeletedcons != NULL);
   assert(ndeletedvars != NULL);

   *ndeletedcons = 0;
   *ndeletedvars = 0;

   /* fix variables */
   for( i = 0; i < nvarstofix; i++ )
   {
      SCIP_CALL( SCIPfixVar(scip, varstofix[i], varsfixvalues[i], &infeasible, &fixed) );
      assert(!infeasible);
      assert(fixed);
      (*ndeletedvars)++;
   }

   /* delete constraints */
   for( i = 0; i < nconsstodelete; ++i )
   {
      SCIP_CALL( SCIPdelCons(scip, consstodelete[i]) );
      (*ndeletedcons)++;
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
   SCIP_DIGRAPH* digraph;
   SCIP_CONS** conss;
   SCIP_CONS** tmpconss;
   int* firstvaridxpercons;
   int nconss;
   int ntmpconss;
   int nvars;
   int* components;
   int ncomponents;
   int ndeletedcons;
   int ndeletedvars;
   int nsolvedprobs;
   int c;
   SCIP_Bool success;

   SCIP_CONS** constodelete;
   int nconstodelete;
   SCIP_VAR** varstofix;
   SCIP_Real* varsfixvalues;
   int nvarstofix;

   int statistics[4] = {0,0,0,0};

   assert(scip != NULL);
   assert(presol != NULL);
   assert(result != NULL);

   *result = SCIP_DIDNOTRUN;

   if( SCIPgetStage(scip) != SCIP_STAGE_PRESOLVING || SCIPinProbing(scip) )
      return SCIP_OKAY;

   if( SCIPgetNActivePricers(scip) > 0 )
      return SCIP_OKAY;

   presoldata = SCIPpresolGetData(presol);
   assert(presoldata != NULL);
   if( !presoldata->dosearch || presoldata->didsearch )
   {
      /* do not search for components */
      return SCIP_OKAY;
   }

   *result = SCIP_DIDNOTFIND;
   presoldata->didsearch = TRUE;

   ncomponents = 0;
   ndeletedvars = 0;
   ndeletedcons = 0;
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

   nvars = SCIPgetNVars(scip);

   if( nvars > 1 && nconss > 1 )
   {
      SCIP_CALL( SCIPallocBufferArray(scip, &constodelete, nconss) );
      SCIP_CALL( SCIPallocBufferArray(scip, &varstofix, nvars) );
      SCIP_CALL( SCIPallocBufferArray(scip, &varsfixvalues, nvars) );
      SCIP_CALL( SCIPallocBufferArray(scip, &firstvaridxpercons, nconss) );

      nconstodelete = 0;
      nvarstofix = 0;

      /* create and fill directed graph */
      SCIP_CALL( SCIPdigraphCreate(&digraph, nvars) );
      SCIP_CALL( fillDigraph(scip, digraph, conss, nconss, firstvaridxpercons, &success) );

      if( success )
      {
         SCIP_CALL( SCIPallocBufferArray(scip, &components, nvars) );

         /* compute independent components */
         SCIP_CALL( SCIPdigraphComputeComponents(digraph, components, &ncomponents) );

         /* create subproblems from independent components and solve them in dependence on their size */
         SCIP_CALL( splitProblem(scip, presoldata, conss, nconss, components, &ncomponents, firstvaridxpercons,
               &nsolvedprobs, &nconstodelete, constodelete, &nvarstofix, varstofix, varsfixvalues, statistics, result) );

         /* fix variables and delete constraints of solved subproblems */
         SCIP_CALL( fixVarsDeleteConss(scip, nconstodelete, constodelete,
                nvarstofix, varstofix, varsfixvalues, &ndeletedcons, &ndeletedvars) );

         (*nfixedvars) += ndeletedvars;
         (*ndelconss) += ndeletedcons;

         SCIPfreeBufferArray(scip, &components);
      }

      SCIPdigraphFree(&digraph);

      SCIPfreeBufferArray(scip, &firstvaridxpercons);
      SCIPfreeBufferArray(scip, &varsfixvalues);
      SCIPfreeBufferArray(scip, &varstofix);
      SCIPfreeBufferArray(scip, &constodelete);
   }

   SCIPfreeBufferArray(scip, &conss);

   if( (ndeletedvars > 0 || ndeletedcons > 0) && ((*result) == SCIP_DIDNOTFIND) )
      *result = SCIP_SUCCESS;

   SCIPdebugMessage("### %d comp (distribution: [1-20]=%d, [21-50]=%d, [51-100]=%d, >100=%d), %d solved, %d delcons, %d delvars\n",
      ncomponents, statistics[0], statistics[1], statistics[2], statistics[3], nsolvedprobs, ndeletedcons, ndeletedvars);

   return SCIP_OKAY;
}


/*
 * Callback methods of presolver
 */

/** copy method for constraint handler plugins (called when SCIP copies plugins) */
#if 0
static
SCIP_DECL_PRESOLCOPY(presolCopyComponents)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of components presolver not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define presolCopyComponents NULL
#endif


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

/* define unused callbacks as NULL */
#define presolInitComponents NULL
#define presolExitComponents NULL
#define presolInitpreComponents NULL
#define presolExitpreComponents NULL


/** execution method of presolver */
static
SCIP_DECL_PRESOLEXEC(presolExecComponents)
{  /*lint --e{715}*/
   *result = SCIP_DIDNOTRUN;

   SCIPdebugMessage("presolExecComponents(): SCIPisPresolveFinished() = %d\n", SCIPisPresolveFinished(scip));

   /* only call the component presolver, if presolving would be stopped otherwise */
   if( SCIPisPresolveFinished(scip) )
   {
      SCIP_CALL( presolComponents(scip, presol, nfixedvars, ndelconss, result) );
   }

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

   /* create components presolver data */
   SCIP_CALL( SCIPallocMemory(scip, &presoldata) );
   presoldata->didsearch = FALSE;

   /* include presolver */
   SCIP_CALL( SCIPincludePresol(scip,
         PRESOL_NAME,
         PRESOL_DESC,
         PRESOL_PRIORITY,
         PRESOL_MAXROUNDS,
         PRESOL_DELAY,
         presolCopyComponents,
         presolFreeComponents,
         presolInitComponents,
         presolExitComponents,
         presolInitpreComponents,
         presolExitpreComponents,
         presolExecComponents,
         presoldata) );

   /* add presolver parameters */
   SCIP_CALL( SCIPaddBoolParam(scip,
         "presolving/components/dosearch",
         "search for components (0: no search, 1: do search)",
         &presoldata->dosearch, FALSE, DEFAULT_SEARCH, NULL, NULL) );
   SCIP_CALL( SCIPaddBoolParam(scip,
         "presolving/components/writeproblems",
         "should the single components be written as an .lp-file?",
         &presoldata->writeproblems, FALSE, DEFAULT_WRITEPROBLEMS, NULL, NULL) );
   SCIP_CALL( SCIPaddIntParam(scip,
         "presolving/components/maxintvars",
         "maximum number of integer (or binary) variables to solve a subproblem directly (-1: no solving)",
         &presoldata->maxintvars, FALSE, DEFAULT_MAXINTVARS, -1, INT_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddLongintParam(scip,
         "presolving/components/nodelimit",
         "maximum number of nodes to be solved in subproblems",
         &presoldata->nodelimit, FALSE, DEFAULT_NODELIMIT, -1, SCIP_LONGINT_MAX, NULL, NULL) );
   SCIP_CALL( SCIPaddRealParam(scip,
         "presolving/components/intfactor",
         "the weight of an integer variable compared to binary variables",
         &presoldata->intfactor, FALSE, DEFAULT_INTFACTOR, 0, SCIP_REAL_MAX, NULL, NULL) );

   return SCIP_OKAY;
}
