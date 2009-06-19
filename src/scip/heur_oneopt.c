/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2009 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not email to scip@zib.de.      */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
#pragma ident "@(#) $Id: heur_oneopt.c,v 1.15.2.1 2009/06/19 07:53:43 bzfwolte Exp $"

/**@file   heur_oneopt.c
 * @ingroup PRIMALHEURISTICS
 * @brief  oneopt primal heuristic
 * @author Timo Berthold
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include <string.h>

#include "scip/heur_oneopt.h"

#define HEUR_NAME             "oneopt"
#define HEUR_DESC             "1-opt heuristic which tries to improve setting of single integer variables"
#define HEUR_DISPCHAR         'k'
#define HEUR_PRIORITY         -20000
#define HEUR_FREQ             1
#define HEUR_FREQOFS          0
#define HEUR_MAXDEPTH         -1
#define HEUR_TIMING           SCIP_HEURTIMING_AFTERNODE

#define DEFAULT_WEIGHTEDOBJ   TRUE


/*
 * Data structures
 */

/** primal heuristic data */
struct SCIP_HeurData
{
   SCIP_Longint nodenumber;
   SCIP_Bool weightedobj;
};




/*
 * Local methods
 */

static
SCIP_Real calcShiftVal(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_VAR*             var,                /**< variable that should be shifted */
   SCIP_Real             solval,             /**< current solution value */
   SCIP_Real*            activities
   )
{
   SCIP_Real lb;
   SCIP_Real ub;
   SCIP_Real obj;
   SCIP_Real shiftval;

   SCIP_COL* col;
   SCIP_ROW** colrows;
   SCIP_Real* colvals;

   int ncolrows;
   int i;

   SCIP_Bool shiftdown;

   /* get variable's solution value, global bounds and objective coefficient */
   lb = SCIPvarGetLbGlobal(var);
   ub = SCIPvarGetUbGlobal(var);
   obj = SCIPvarGetObj(var);
   shiftval = 0.0;
   shiftdown = TRUE;

   /* determine shifting direction and maximal possible shifting wrt. corresponding bound */
   if( obj > 0.0 && SCIPisFeasGE(scip, solval - 1.0, lb) )
      shiftval = SCIPfeasFloor(scip, solval - lb);
   else if ( obj < 0.0 && SCIPisFeasLE(scip, solval + 1.0, ub) )
   {
      shiftval = SCIPfeasFloor(scip, ub - solval);
      shiftdown = FALSE;
   }
   else
      return 0.0;


   SCIPdebugMessage("Try to shift %s variable <%s> with\n", shiftdown ? "down" : "up", SCIPvarGetName(var) );
   SCIPdebugMessage("    lb:<%g> <= val:<%g> <= ub:<%g> and obj:<%g> by at most: <%g>\n", lb, solval, ub, obj, shiftval);

   /* get data of LP column */
   col = SCIPvarGetCol(var);
   colrows = SCIPcolGetRows(col);
   colvals = SCIPcolGetVals(col);
   ncolrows = SCIPcolGetNLPNonz(col);
   assert(ncolrows == 0 || (colrows != NULL && colvals != NULL));

   /* find minimal shift value, st. all rows stay valid */
   for( i = 0; i < ncolrows && shiftval > 0.0; ++i )
   {
      SCIP_ROW* row;
      int rowpos;

      row = colrows[i];
      rowpos = SCIProwGetLPPos(row);
      assert(-1 <= rowpos && rowpos < SCIPgetNLPRows(scip) );

      /* only global rows need to be valid */
      if( rowpos >= 0 && !SCIProwIsLocal(row) )
      {
         SCIP_Real shiftvalrow;

         assert(SCIProwIsInLP(row));

         if( shiftdown == (colvals[i] > 0) )
            shiftvalrow = SCIPfeasFloor(scip, (activities[rowpos] - SCIProwGetLhs(row)) / ABS(colvals[i]));
         else
            shiftvalrow = SCIPfeasFloor(scip, (SCIProwGetRhs(row) -  activities[rowpos]) / ABS(colvals[i]));
#ifdef SCIP_DEBUG
         if( shiftvalrow < shiftval )
         {
            SCIPdebugMessage(" -> The shift value had to be reduced to <%g>, because of row <%s>.\n",
               shiftvalrow, SCIProwGetName(row));
            SCIPdebugMessage("    lhs:<%g> <= act:<%g> <= rhs:<%g>, colval:<%g>\n",
               SCIProwGetLhs(row), activities[rowpos], SCIProwGetRhs(row), colvals[i]);
         }
#endif
         shiftval = MIN(shiftval, shiftvalrow);
      }
   }
   if( shiftdown )
      shiftval *= -1.0;

   return shiftval;
}


/** update row activities after a variable's solution value changed */
static
SCIP_RETCODE updateRowActivities(
   SCIP*                 scip,               /**< SCIP data structure */
   SCIP_Real*            activities,         /**< LP row activities */
   SCIP_VAR*             var,                /**< variable that has been changed */
   SCIP_Real             shiftval            /**< value that is added to variable */
   )
{
   SCIP_Real* colvals;
   SCIP_ROW** colrows;
   SCIP_COL* col;

   int ncolrows;
   int i;

   assert(activities != NULL);

   /* get data of column associated to variable */
   col = SCIPvarGetCol(var);
   colrows = SCIPcolGetRows(col);
   colvals = SCIPcolGetVals(col);
   ncolrows = SCIPcolGetNLPNonz(col);
   assert(ncolrows == 0 || (colrows != NULL && colvals != NULL));

   /* enumerate all rows with nonzero entry in this column */
   for( i = 0; i < ncolrows; ++i )
   {
      SCIP_ROW* row;
      int rowpos;

      row = colrows[i];
      rowpos = SCIProwGetLPPos(row);
      assert(-1 <= rowpos && rowpos < SCIPgetNLPRows(scip) );

      /* update row activity, only regard global rows in the LP */
      if( rowpos >= 0 && !SCIProwIsLocal(row) )
         activities[rowpos] +=  shiftval * colvals[i];

      if( SCIPisInfinity(scip, activities[rowpos]) )
         activities[rowpos] = SCIPinfinity(scip);
      else if( SCIPisInfinity(scip, -activities[rowpos]) )
         activities[rowpos] = -SCIPinfinity(scip);
   }

   return SCIP_OKAY;
}

/*
 * Callback methods of primal heuristic
 */

/** destructor of primal heuristic to free user data (called when SCIP is exiting) */
static
SCIP_DECL_HEURFREE(heurFreeOneopt)
{   /*lint --e{715}*/
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
#define heurInitOneopt NULL

/** deinitialization method of primal heuristic (called before transformed problem is freed) */
#define heurExitOneopt NULL

/** solving process initialization method of primal heuristic (called when branch and bound process is about to begin) */
static
SCIP_DECL_HEURINITSOL(heurInitsolOneopt)
{
   SCIP_HEURDATA* heurdata;

   assert(strcmp(SCIPheurGetName(heur), HEUR_NAME) == 0);

   /* create heuristic data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);
   heurdata->nodenumber = -1;

   return SCIP_OKAY;
}

/** solving process deinitialization method of primal heuristic (called before branch and bound process data is freed) */
#define heurExitsolOneopt NULL

/** execution method of primal heuristic */
static
SCIP_DECL_HEUREXEC(heurExecOneopt)
{  /*lint --e{715}*/

   SCIP_HEURDATA* heurdata;
   SCIP_SOL* bestsol;                        /* incumbent solution */
   SCIP_SOL* worksol;                        /* heuristic's working solution */

   SCIP_VAR** vars;                          /* SCIP variables                */
   SCIP_VAR** shiftcands;                    /* shiftable variables           */
   SCIP_ROW** lprows;                        /* SCIP LP rows                  */
   SCIP_Real* activities;                    /* row activities for working solution */
   SCIP_Real* shiftvals;

   int nbinvars;
   int nintvars;
   int nvars;
   int nlprows;
   int i;
   int nshiftcands;
   int shiftcandssize;

   assert(heur != NULL);
   assert(scip != NULL);
   assert(result != NULL);

   /* get heuristic's data */
   heurdata = SCIPheurGetData(heur);
   assert(heurdata != NULL);

   *result = SCIP_DELAYED;

   /* we need an LP */
   if( SCIPgetNLPRows(scip) == 0 )
      return SCIP_OKAY;

   /* we only want to process each solution once */
   bestsol = SCIPgetBestSol(scip);
   if( bestsol == NULL  || heurdata->nodenumber == SCIPgetSolNodenum(scip, bestsol) )
      return SCIP_OKAY;

   /* we can only work on solutions valid in the transformed space */
   if( SCIPsolGetOrigin(bestsol) == SCIP_SOLORIGIN_ORIGINAL )
      return SCIP_OKAY;

   /* get problem variables */
   SCIP_CALL( SCIPgetVarsData(scip, &vars, &nvars, &nbinvars, &nintvars, NULL, NULL) );
   nintvars = nbinvars + nintvars;

   /* we need to be able to start diving from current node in order to resolve the LP
    * with continuous or implicit integer variables
    */
   if( nvars > nintvars )
   {
      if( !SCIPhasCurrentNodeLP(scip) || SCIPgetLPSolstat(scip) != SCIP_LPSOLSTAT_OPTIMAL )
	 return SCIP_OKAY;
   }

   *result = SCIP_DIDNOTFIND;

   /* initialize data */
   nshiftcands = 0;
   shiftcandssize = 8;
   heurdata->nodenumber = SCIPgetSolNodenum(scip, bestsol);

   SCIP_CALL( SCIPcreateSolCopy(scip, &worksol, bestsol) );
   SCIPsolSetHeur(worksol,heur);

   SCIP_CALL( SCIPgetLPRowsData(scip, &lprows, &nlprows) );
   SCIP_CALL( SCIPallocBufferArray(scip, &activities, nlprows) );
   SCIP_CALL( SCIPallocBufferArray(scip, &shiftcands, shiftcandssize) );
   SCIP_CALL( SCIPallocBufferArray(scip, &shiftvals, shiftcandssize) );

   /* initialize activities */
   for( i = 0; i < nlprows; ++i )
   {
      SCIP_ROW* row;

      row = lprows[i];
      assert(SCIProwGetLPPos(row) == i);

      if( !SCIProwIsLocal(row) )
      {
         activities[i] = SCIPgetRowSolActivity(scip, row, bestsol);
         SCIPdebugMessage("Row <%s> has activity %g\n", SCIProwGetName(row), activities[i]);
      }
   }

   SCIPdebugMessage("Starting 1-opt heuristic\n");

   /* enumerate all integer variables and find out which of them are shiftable */
   for( i = 0; i < nintvars; i++ )
   {
      if( SCIPvarGetStatus(vars[i]) == SCIP_VARSTATUS_COLUMN )
      {
         SCIP_Real shiftval;
         SCIP_Real solval;

         /* find out whether the variable can be shifted */
         solval = SCIPgetSolVal(scip, bestsol,vars[i]);
         shiftval = calcShiftVal(scip, vars[i], solval, activities);

         /* insert the variable into the list of shifting candidates */
         if( !SCIPisFeasZero(scip, shiftval) )
         {
            SCIPdebugMessage(" -> Variable <%s> can be shifted by <%1.1f> \n", SCIPvarGetName(vars[i]), shiftval);

            if( nshiftcands == shiftcandssize)
            {
               shiftcandssize *= 8;
               SCIP_CALL( SCIPreallocBufferArray(scip, &shiftcands, shiftcandssize) );
               SCIP_CALL( SCIPreallocBufferArray(scip, &shiftvals, shiftcandssize) );
            }
            shiftcands[nshiftcands] = vars[i];
            shiftvals[nshiftcands] = shiftval;
            nshiftcands++;
         }
      }
   }

   /* if at least one variable can be shifted, shift variables sorted by their objective */
   if( nshiftcands > 0 )
   {
      SCIP_Real shiftval;
      SCIP_Real solval;
      SCIP_VAR* var;

      /* the case that exactly one variable can be shifted is slightly easier */
      if( nshiftcands == 1 )
      {
         var = shiftcands[0];
         assert(var != NULL);
         solval = SCIPgetSolVal(scip, bestsol, var);
         shiftval = shiftvals[0];
         assert(!SCIPisFeasZero(scip,shiftval));
         SCIPdebugMessage(" Only one shiftcand found, var <%s>, which is now shifted by<%1.1f> \n",
            SCIPvarGetName(var), shiftval);
         SCIP_CALL( SCIPsetSolVal(scip, worksol, var, solval+shiftval) );
      }
      else
      {
         SCIP_Real* objcoeffs;

         SCIP_CALL( SCIPallocBufferArray(scip, &objcoeffs, nshiftcands) );

         SCIPdebugMessage(" %d shiftcands found \n", nshiftcands);

         /* sort the variables by their objective, optionally weighted with the shiftval */
         if( heurdata->weightedobj )
         {
            for( i = 0; i < nshiftcands; ++i )
               objcoeffs[i] = SCIPvarGetObj(shiftcands[i])*shiftvals[i];
         }
         else
         {
            for( i = 0; i < nshiftcands; ++i )
               objcoeffs[i] = SCIPvarGetObj(shiftcands[i]);
         }

         /* sort arrays with respect to the first one */
         SCIPsortRealPtr(objcoeffs, (void*)shiftcands, nshiftcands);

         /* try to shift each variable -> Activities have to be updated */
         for( i = 0; i < nshiftcands; ++i )
         {
            var = shiftcands[i];
            assert(var != NULL);
            solval = SCIPgetSolVal(scip, bestsol, var);
            shiftval = calcShiftVal(scip, var, solval, activities);
            SCIPdebugMessage(" -> Variable <%s> is now shifted by <%1.1f> \n", SCIPvarGetName(vars[i]), shiftval);
            assert(i > 0 || !SCIPisFeasZero(scip, shiftval));
            SCIP_CALL( SCIPsetSolVal(scip, worksol, var, solval+shiftval) );
            SCIP_CALL( updateRowActivities(scip, activities, var, shiftval) );
         }

         SCIPfreeBufferArray(scip, &objcoeffs);
      }

      /* if the problem is a pure IP, try to install the solution, if it is a MIP, solve LP again to set the continuous
       * variables to the best possible value
       */
      if( nvars == nintvars )
      {
         SCIP_Bool success;

         SCIP_CALL( SCIPtrySol(scip, worksol, FALSE, FALSE, FALSE, &success) );
         if( success )
         {
            SCIPdebugMessage("found feasible shifted solution:\n");
            SCIPdebug(SCIPprintSol(scip, worksol, NULL, FALSE));
            *result = SCIP_FOUNDSOL;
         }
      }
      else
      {
         SCIP_Bool lperror;
#ifdef NDEBUG
         SCIP_RETCODE retstat;
#endif

         SCIPdebugMessage("shifted solution should be feasible -> solve LP to fix continuous variables to best values\n");

         /* start diving to calculate the LP relaxation */
         SCIP_CALL( SCIPstartDive(scip) );

         /* set the bounds of the variables: fixed for integers, global bounds for continuous */
         for( i = 0; i < nvars; ++i )
         {
            if( SCIPvarGetStatus(vars[i]) == SCIP_VARSTATUS_COLUMN )
            {
               SCIP_CALL( SCIPchgVarLbDive(scip, vars[i], SCIPvarGetLbGlobal(vars[i])) );
               SCIP_CALL( SCIPchgVarUbDive(scip, vars[i], SCIPvarGetUbGlobal(vars[i])) );
            }
         }
         /* apply this after global bounds to not cause an error with intermediate empty domains */
         for( i = 0; i < nintvars; ++i )
         {
            if( SCIPvarGetStatus(vars[i]) == SCIP_VARSTATUS_COLUMN )
            {
               solval = SCIPgetSolVal(scip, worksol, vars[i]);
               SCIP_CALL( SCIPchgVarLbDive(scip, vars[i], solval) );
               SCIP_CALL( SCIPchgVarUbDive(scip, vars[i], solval) );
            }
         }

         /* solve LP */
         SCIPdebugMessage(" -> old LP iterations: %"SCIP_LONGINT_FORMAT"\n", SCIPgetNLPIterations(scip));

         /* Errors in the LP solver should not kill the overall solving process, if the LP is just needed for a heuristic.
          * Hence in optimized mode, the return code is catched and a warning is printed, only in debug mode, SCIP will stop.
          */
#ifdef NDEBUG
         retstat = SCIPsolveDiveLP(scip, -1, &lperror);
         if( retstat != SCIP_OKAY )
         { 
            SCIPwarningMessage("Error while solving LP in Oneopt heuristic; LP solve terminated with code <%d>\n",retstat);
         }
#else
         SCIP_CALL( SCIPsolveDiveLP(scip, -1, &lperror) );
#endif
         
         SCIPdebugMessage(" -> new LP iterations: %"SCIP_LONGINT_FORMAT"\n", SCIPgetNLPIterations(scip));
         SCIPdebugMessage(" -> error=%d, status=%d\n", lperror, SCIPgetLPSolstat(scip));

         /* check if this is a feasible solution */
         if( !lperror && SCIPgetLPSolstat(scip) == SCIP_LPSOLSTAT_OPTIMAL )
         {
            SCIP_Bool success;
            /* copy the current LP solution to the working solution */
            SCIP_CALL( SCIPlinkLPSol(scip, worksol) );

            /* check solution for feasibility */
            SCIP_CALL( SCIPtrySol(scip, worksol, FALSE, FALSE, FALSE, &success) );

            if( success )
            {
               SCIPdebugMessage("found feasible shifted solution:\n");
               SCIPdebug(SCIPprintSol(scip, worksol, NULL, FALSE));
               *result = SCIP_FOUNDSOL;
            }
         }

         /* terminate the diving */
         SCIP_CALL( SCIPendDive(scip) );
      }
   }
   SCIPdebugMessage("Finished 1-opt heuristic\n");

   SCIP_CALL( SCIPfreeSol(scip, &worksol) );
   SCIPfreeBufferArray(scip, &activities);
   SCIPfreeBufferArray(scip, &shiftvals);
   SCIPfreeBufferArray(scip, &shiftcands);

   return SCIP_OKAY;
}





/*
 * primal heuristic specific interface methods
 */

/** creates the oneopt primal heuristic and includes it in SCIP */
SCIP_RETCODE SCIPincludeHeurOneopt(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_HEURDATA* heurdata;

   /* create heuristic data */
   SCIP_CALL( SCIPallocMemory(scip, &heurdata) );

   /* include primal heuristic */
   SCIP_CALL( SCIPincludeHeur(scip, HEUR_NAME, HEUR_DESC, HEUR_DISPCHAR, HEUR_PRIORITY, HEUR_FREQ, HEUR_FREQOFS,
         HEUR_MAXDEPTH, HEUR_TIMING,
         heurFreeOneopt, heurInitOneopt, heurExitOneopt,
         heurInitsolOneopt, heurExitsolOneopt, heurExecOneopt,
         heurdata) );

   /* add oneopt primal heuristic parameters */

   SCIP_CALL( SCIPaddBoolParam(scip, "heuristics/oneopt/weightedobj",
         "should the objective be weighted with the potential shifting value when sorting the shifting candidates?",
         &heurdata->weightedobj, TRUE, DEFAULT_WEIGHTEDOBJ, NULL, NULL) );

   return SCIP_OKAY;
}
