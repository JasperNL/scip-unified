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

#define START_LIST_SIZE              10      /**< first size of constraint list per variable */
#define LIST_MEMORY_GAIN             10      /**< memory extension factor */

/*
 * Data structures
 */

/** control parameters */
struct SCIP_PresolData
{
   SCIP_Bool             dosearch;           /** should be searched for components? */
   SCIP_Bool             writeproblems;      /** should the single components be written as an .lp-file? */
   int                   maxintvars;         /** maximum number of integer (or binary) variables to solve a subproblem directly (-1: no solving) */
   SCIP_Longint          nodelimit;          /** maximum number of nodes to be solved in subproblems */
   SCIP_Real             intfactor;          /** the weight of an integer variable compared to binary variables */

   SCIP**                components;         /** sub-SCIPs storing the components */
   SCIP_HASHMAP**        varmaps;            /** hashmaps mapping from original variables to variables in the sub-SCIPs */
   SCIP_HASHMAP*         consmap;            /** hashmaps mapping from original constraints to constraints in the sub-SCIPs
                                              *  (needed only for performance reasons)
                                              */
   int                   componentssize;     /** size of arrays components and varmaps */
   int                   ncomponents;        /** number of components */
};

/*
 * Local methods
 */

/** comparison method for two sorting key pairs */
static
SCIP_DECL_SORTPTRCOMP(elementComparator)
{
   int* key1 = (int*)elem1;
   int* key2 = (int*)elem2;

   if( *key1 < *key2 )
      return -1;
   else if( *key1 > *key2 )
      return +1;
   else
      return 0;
}

/** initializes presolver data */
static
void initPresoldata(
   SCIP_PRESOLDATA*      presoldata          /**< presolver data */
   )
{
   assert(presoldata != NULL);

   presoldata->dosearch = 0;
   presoldata->components = NULL;
   presoldata->varmaps = NULL;
   presoldata->consmap = NULL;
   presoldata->componentssize = 0;
   presoldata->ncomponents = 0;
}

/** initializes the data for storing connected components */
static
SCIP_RETCODE initComponentData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOLDATA*      presoldata,         /**< presolver data */
   int                   ncomponents         /**< number of independent components */
   )
{
   assert(scip != NULL);
   assert(presoldata != NULL);
   assert(ncomponents > 0);
   assert(presoldata->ncomponents == 0);
   assert(presoldata->componentssize == 0);
   assert(presoldata->components == NULL);
   assert(presoldata->varmaps == NULL);
   assert(presoldata->consmap == NULL);

   /* allocate memory for sub-SCIPs and variable maps */
   SCIP_CALL( SCIPallocMemoryArray(scip, &presoldata->components, ncomponents) );
   SCIP_CALL( SCIPallocMemoryArray(scip, &presoldata->varmaps, ncomponents) );
   SCIP_CALL( SCIPhashmapCreate(&presoldata->consmap, SCIPblkmem(scip), 10 * SCIPgetNConss(scip)) );
   presoldata->componentssize = ncomponents;

   return SCIP_OKAY;
}

/** free the data for storing connected components */
static
SCIP_RETCODE freeComponentData(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOLDATA*      presoldata          /**< presolver data */
   )
{
   int c;

   assert(scip != NULL);
   assert(presoldata != NULL);

   /* free sub-SCIPs and variable hash maps */
   for( c = 0; c < presoldata->ncomponents; ++c )
   {
      if( presoldata->components[c] != NULL )
      {
         SCIP_CALL( SCIPfree(&presoldata->components[c]) );
      }
      if( presoldata->varmaps[c] != NULL )
      {
         SCIPhashmapFree(&presoldata->varmaps[c]);
      }
   }

   SCIPhashmapFree(&presoldata->consmap);

   SCIPfreeMemoryArray(scip, &presoldata->components);
   SCIPfreeMemoryArray(scip, &presoldata->varmaps);
   presoldata->ncomponents = 0;
   presoldata->componentssize = 0;
   presoldata->components = NULL;
   presoldata->varmaps = NULL;

   return SCIP_OKAY;
}

/** copies a connected component given by a set of constraints into a sub-SCIP */
static
SCIP_RETCODE buildComponentSubscip(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_PRESOLDATA*      presoldata,         /**< presolver data */
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
   SCIP_Real*            varsfixvalues       /**< fixing values of the variables */
   )
{
   char probname[SCIP_MAXSTRLEN];
   char consname[SCIP_MAXSTRLEN];
   SCIP* subscip;
   SCIP_CONS* newcons;
   SCIP_Real timelimit;
   SCIP_Real memorylimit;
   SCIP_Bool success;
   int c;
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

   c = presoldata->ncomponents;

   //SCIPdebugMessage("build sub-SCIP for component %d (%d vars, %d conss)\n", c, nvars, nconss);

   /* create sub-SCIP */
   SCIP_CALL( SCIPcreate(&(presoldata->components[c])) );
   subscip = presoldata->components[c];

   /* create variable hashmap */
   SCIP_CALL( SCIPhashmapCreate(&presoldata->varmaps[c], SCIPblkmem(scip), 10 * nvars) );

   /* copy plugins */
   success = TRUE;
   SCIP_CALL( SCIPcopyPlugins(scip, subscip,
         TRUE, /* readers */
         TRUE, /* pricers */
         TRUE, /* conshdlrs */
         TRUE, /* conflicthdlrs */
         TRUE, /* presolvers */
         TRUE, /* relaxators */
         TRUE, /* separators */
         TRUE, /* propagators */
         TRUE, /* heuristics */
         TRUE, /* eventhandler */
         TRUE, /* nodeselectors (SCIP gives an error if there is none) */
         TRUE, /* branchrules */
         TRUE, /* displays */
         FALSE, /* dialogs */
         TRUE, /* nlpis */
         &success) );

   if( success )
   {
      /* copy parameter settings */
      SCIP_CALL( SCIPcopyParamSettings(scip, subscip) );

#if 1
      /* reduce the effort spent for hash tables */
      SCIP_CALL( SCIPsetBoolParam(subscip, "misc/usevartable", FALSE) );
      SCIP_CALL( SCIPsetBoolParam(subscip, "misc/useconstable", FALSE) );
      SCIP_CALL( SCIPsetBoolParam(subscip, "misc/usesmalltables", TRUE) );
#endif

      /* set gap limit to 0 */
      SCIP_CALL( SCIPsetRealParam(subscip, "limits/gap", 0.0) );

      /* do not catch control-C */
      SCIP_CALL( SCIPsetBoolParam(subscip, "misc/catchctrlc", FALSE) );


      /* check whether there is enough time and memory left */
      timelimit = 0.0;
      memorylimit = 0.0;
      SCIP_CALL( SCIPgetRealParam(scip, "limits/time", &timelimit) );
      if( !SCIPisInfinity(scip, timelimit) )
         timelimit -= SCIPgetSolvingTime(scip);
      SCIP_CALL( SCIPgetRealParam(scip, "limits/memory", &memorylimit) );
      if( !SCIPisInfinity(scip, memorylimit) )
         memorylimit -= SCIPgetMemUsed(scip)/1048576.0;
      if( timelimit <= 0.0 || memorylimit <= 0.0 )
         goto TERMINATE;

      /* set limits for the subproblem */
      SCIP_CALL( SCIPsetRealParam(subscip, "limits/time", timelimit) );
      SCIP_CALL( SCIPsetRealParam(subscip, "limits/memory", memorylimit) );

      /* set node limit */
      if( presoldata->nodelimit != -1 )
      {
         SCIP_CALL( SCIPsetLongintParam(subscip, "limits/nodes", presoldata->nodelimit) );
      }
      //SCIP_CALL( SCIPsetLongintParam(subscip, "limits/stallnodes", 20000) );

      /* disable output */
      SCIP_CALL( SCIPsetIntParam(subscip, "display/verblevel", SCIP_VERBLEVEL_NONE) );

      /* create problem in sub-SCIP */
      /* get name of the original problem and add "comp_nr" */
      (void) SCIPsnprintf(probname, SCIP_MAXSTRLEN, "%s_comp_%d", SCIPgetProbName(scip), c);
      SCIP_CALL( SCIPcreateProb(subscip, probname, NULL, NULL, NULL, NULL, NULL, NULL, NULL) );

      for( i = 0; i < nconss; ++i )
      {
         /* copy the constraint */
         (void) SCIPsnprintf(consname, SCIP_MAXSTRLEN, "%s", SCIPconsGetName(conss[i]));
         SCIP_CALL( SCIPgetConsCopy(scip, subscip, conss[i], &newcons, SCIPconsGetHdlr(conss[i]),
               presoldata->varmaps[c], presoldata->consmap, consname,
               TRUE, TRUE, TRUE, TRUE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE, &success) );

         /* break if constraint was not successfully copied */
         if( !success )
            break;

         SCIP_CALL( SCIPaddCons(subscip, newcons) );
         SCIP_CALL( SCIPreleaseCons(subscip, &newcons) );
      }
   }

   /* ignore this component, if a problem relevant plugin or a constraint could not be copied */
   if( success )
   {
      presoldata->ncomponents++;

      //printf("++++++++++++++ sub-SCIP for component %d: %d vars (%d bin, %d int, %d impl, %d cont), %d conss\n",
      //   c, nvars, SCIPgetNBinVars(subscip), SCIPgetNIntVars(subscip), SCIPgetNImplVars(subscip), SCIPgetNContVars(subscip), nconss);

      if( presoldata->writeproblems )
      {
         (void) SCIPsnprintf(probname, SCIP_MAXSTRLEN, "%s_comp_%d.lp", SCIPgetProbName(scip), c);
         SCIPdebugMessage("write problem to file %s\n", probname);
         SCIP_CALL( SCIPwriteOrigProblem(subscip, probname, NULL, FALSE) );
      }

      if( SCIPgetNBinVars(subscip) + SCIPgetNIntVars(subscip) <= presoldata->maxintvars )
      {
         //SCIP_CALL( SCIPpresolve(subscip) ); // TODO: simulation of presolving without solve

         SCIP_CALL( SCIPsolve(subscip) );

         //printf("solved subproblem %d: status = %d, time = %.2f\n", c, SCIPgetStatus(subscip), SCIPgetSolvingTime(subscip));
         *subsolvetime += SCIPgetSolvingTime(subscip);

         if( SCIPgetStatus(subscip) == SCIP_STATUS_OPTIMAL )
         {
            SCIP_SOL* sol;

            ++(*nsolvedprobs);

            sol = SCIPgetBestSol(subscip);

            /* memorize variables for later fixing */
            for( i = 0; i < nvars; ++i )
            {
               assert( SCIPhashmapExists(presoldata->varmaps[c], vars[i]) );
               varstofix[*nvarstofix] = vars[i];
               varsfixvalues[*nvarstofix] = SCIPgetSolVal(subscip, sol, SCIPhashmapGetImage(presoldata->varmaps[c], vars[i]));
               (*nvarstofix)++;
            }

            /* memorize constraints for later deletion */
            for( i = 0; i < nconss; ++i )
            {
               constodelete[*nconstodelete] = conss[i];
               (*nconstodelete)++;
            }
         }
         else
         {
            SCIPdebugMessage("++++++++++++++ sub-SCIP for component %d not solved (status=%d, time=%.2f): %d vars (%d bin, %d int, %d impl, %d cont), %d conss\n",
               c, SCIPgetStatus(subscip), SCIPgetSolvingTime(subscip), nvars, SCIPgetNBinVars(subscip), SCIPgetNIntVars(subscip), SCIPgetNImplVars(subscip), SCIPgetNContVars(subscip), nconss);
         }
      }
      else
      {
         SCIPdebugMessage("++++++++++++++ sub-SCIP for component %d not solved: %d vars (%d bin, %d int, %d impl, %d cont), %d conss\n",
            c, nvars, SCIPgetNBinVars(subscip), SCIPgetNIntVars(subscip), SCIPgetNImplVars(subscip), SCIPgetNContVars(subscip), nconss);
      }
   }

 TERMINATE:
   SCIP_CALL( SCIPfree(&presoldata->components[c]) );
   SCIPhashmapFree(&presoldata->varmaps[c]);
   presoldata->components[c] = NULL;
   presoldata->varmaps[c] = NULL;

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
      SCIP_CALL( SCIPgetConsNVars(scip, conss[c], &nconsvars, success) );

      if( !(*success) )
         break;

      if( nconsvars > nvars )
      {
         nvars = nconsvars;
         SCIP_CALL( SCIPreallocBufferArray(scip, &consvars, nvars) );
      }

      SCIP_CALL( SCIPgetConsVars(scip, conss[c], consvars, nvars, success) );

      if( !(*success) )
         break;

      SCIP_CALL( SCIPgetActiveVars(scip, consvars, &nconsvars, nvars, &requiredsize) );
      assert(requiredsize <= nvars);

      idx1 = SCIPvarGetProbindex(consvars[0]);
      assert(idx1 >= 0);

      /* save problem index of the first variable for later component assignment */
      firstvaridxpercons[c] = idx1;

      if( nconsvars > 1 )
      {
         /* create sparse directed graph
            sparse means, to add only those edges necessary for component calculation */
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

/* calculate frequency distribution of component sizes */
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

/** use components to detect which vars and cons belong to one subscip
 *  and try to solve all subscips having not too much integer variables
 */
static
SCIP_RETCODE createSubScipsAndSolve(
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
   int*                  statistics          /**< array holding some statistical information */
  )
{
   int comp;
   int v;
   int c;
   SCIP_VAR** vars;
   int nvars;
   SCIP_CONS** tmpconss;
   int ntmpconss;
   SCIP_VAR** tmpvars;
   int ntmpvars;
   int nbinvars;
   int nintvars;
   SCIP_Real subsolvetime;
   int** conscomponent;
   int* considx;
   int** varscomponent;
   int* varsidx;

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

   *nsolvedprobs = 0;
   subsolvetime = 0.0;

   nvars = SCIPgetNVars(scip);
   vars = SCIPgetVars(scip);

   initComponentData(scip, presoldata, *ncomponents);

   SCIP_CALL( SCIPallocBufferArray(scip, &tmpconss, nconss) );
   SCIP_CALL( SCIPallocBufferArray(scip, &tmpvars, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &conscomponent, nconss) );
   SCIP_CALL( SCIPallocBufferArray(scip, &considx, nconss) );
   SCIP_CALL( SCIPallocBufferArray(scip, &varscomponent, nvars) );
   SCIP_CALL( SCIPallocBufferArray(scip, &varsidx, nvars) );

   /* do the mapping from calculated components per variable to corresponding
      constraints and sort the component-arrays for faster finding the
      actual variables and constraints belonging to one component */
   for( c = 0; c < nconss; c++ )
   {
      conscomponent[c] = &components[firstvaridxpercons[c]];
      considx[c] = c;
   }
   for( v = 0; v < nvars; v++ )
   {
      varscomponent[v] = &components[v];
      varsidx[v] = v;
   }
   SCIPsortPtrInt((void**)varscomponent, varsidx, elementComparator, nvars);
   SCIPsortPtrInt((void**)conscomponent, considx, elementComparator, nconss);

   v = 0;
   c = 0;

   /* loop over all components
      start loop from 1 because components are numbered form 1..n */
   for( comp = 1; comp <= *ncomponents; comp++ )
   {
      ntmpconss = 0;
      ntmpvars = 0;
      nbinvars = 0;
      nintvars = 0;

      /* get variables present in this component */
      while( v < nvars && *varscomponent[v] == comp )
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
      while( c < nconss && *conscomponent[c] == comp )
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
         if( ntmpconss > 0 && ntmpvars > 0 )
         {
            /* build subscip for one component and try to solve it */
            SCIP_CALL( buildComponentSubscip(scip, presoldata, tmpconss, ntmpconss,
                  tmpvars, ntmpvars, nsolvedprobs, &subsolvetime, nconstodelete, constodelete,
                  nvarstofix, varstofix, varsfixvalues ) );
         }
         else
         {
            /* this can occur if we have a variable present within
               the obj function but not present within any constraint */
            SCIPdebugMessage("++++++++++++++ sub-SCIP for empty (!) component %d not created: %d vars (%d bin, %d int, %d cont), %d conss\n",
            presoldata->ncomponents, ntmpvars, nbinvars, nintvars, ntmpvars - nintvars - nbinvars, ntmpconss);
         }
      }
      else
      {
         SCIPdebugMessage("++++++++++++++ sub-SCIP for component %d not created: %d vars (%d bin, %d int, %d cont), %d conss\n",
            presoldata->ncomponents, ntmpvars, nbinvars, nintvars, ntmpvars - nintvars - nbinvars, ntmpconss);
      }
   }

   SCIPfreeBufferArray(scip, &varsidx);
   SCIPfreeBufferArray(scip, &varscomponent);
   SCIPfreeBufferArray(scip, &considx);
   SCIPfreeBufferArray(scip, &conscomponent);
   SCIPfreeBufferArray(scip, &tmpvars);
   SCIPfreeBufferArray(scip, &tmpconss);

   freeComponentData(scip, presoldata);

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
   SCIP_RESULT*          result              /**< pointer to store the result of the presolving call */
   )
{
   SCIP_CONS** conss;
   int nconss;
   SCIP_CONS** tmpconss;
   int ntmpconss;
   int nvars;
   SCIP_PRESOLDATA* presoldata;
   SCIP_DIGRAPH* digraph;
   int* components;
   int ncomponents;
   int ndeletedcons;
   int ndeletedvars;
   int nsolvedprobs;
   int c;
   SCIP_Bool success;
   int nconstodelete;
   SCIP_CONS** constodelete;
   int nvarstofix;
   SCIP_VAR** varstofix;
   SCIP_Real* varsfixvalues;
   int* firstvaridxpercons;
   int statistics[4] = {0,0,0,0};

   assert(scip != NULL);
   assert(presol != NULL);
   assert(result != NULL);

   *result = SCIP_DIDNOTRUN;

   if( SCIPgetStage(scip) != SCIP_STAGE_PRESOLVING || SCIPinProbing(scip) )
      return SCIP_OKAY;

   presoldata = SCIPpresolGetData(presol);
   assert(presoldata != NULL);
   if( !presoldata->dosearch )
   {
      /* do not search for components */
      return SCIP_OKAY;
   }

   *result = SCIP_DIDNOTFIND;

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

         /* create subproblems from independent components and solve them */
         SCIP_CALL( createSubScipsAndSolve(scip, presoldata, conss, nconss,
               components, &ncomponents, firstvaridxpercons, &nsolvedprobs, &nconstodelete, constodelete,
               &nvarstofix, varstofix, varsfixvalues, statistics) );

         /* fix variables and delete constraints of solved subproblems */
         SCIP_CALL( fixVarsDeleteConss(scip, nconstodelete, constodelete,
                nvarstofix, varstofix, varsfixvalues, &ndeletedcons, &ndeletedvars) );

         SCIPfreeBufferArray(scip, &components);
      }

      SCIPdigraphFree(&digraph);

      SCIPfreeBufferArray(scip, &firstvaridxpercons);
      SCIPfreeBufferArray(scip, &varsfixvalues);
      SCIPfreeBufferArray(scip, &varstofix);
      SCIPfreeBufferArray(scip, &constodelete);
   }

   SCIPfreeBufferArray(scip, &conss);

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


/** initialization method of presolver (called after problem was transformed) */
#if 0
static
SCIP_DECL_PRESOLINIT(presolInitComponents)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of components presolver not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define presolInitComponents NULL
#endif


/** deinitialization method of presolver (called before transformed problem is freed) */
#if 0
static
SCIP_DECL_PRESOLEXIT(presolExitComponents)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of components presolver not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define presolExitComponents NULL
#endif


/** presolving initialization method of presolver (called when presolving is about to begin) */
#if 0
static
SCIP_DECL_PRESOLINITPRE(presolInitpreComponents)
{  /*lint --e{715}*/
   SCIPerrorMessage("method of components presolver not implemented yet\n");
   SCIPABORT(); /*lint --e{527}*/

   return SCIP_OKAY;
}
#else
#define presolInitpreComponents NULL
#endif


/** presolving deinitialization method of presolver (called after presolving has been finished) */
static
SCIP_DECL_PRESOLEXITPRE(presolExitpreComponents)
{  /*lint --e{715}*/

   SCIP_CALL( presolComponents(scip, presol, result) );

   *result = SCIP_FEASIBLE;

   return SCIP_OKAY;
}


/** execution method of presolver */
static
SCIP_DECL_PRESOLEXEC(presolExecComponents)
{  /*lint --e{715}*/

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
   initPresoldata(presoldata);

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
