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

/**@file   presol_dualagg.c
 * @brief  aggregate variables by dual arguments
 * @author Dieter Weninger
 *
 * This presolver looks for variables which could not be handled by
 * duality fixing because of one violated up-/downlock.
 * If this constraint which delivers the violated up-/downlock has
 * a specific structure, we can aggregate the corresponding variable.
 *
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "scip/scipdefplugins.h"
#include "scip/pub_matrix.h"

#include "presol_dualagg.h"

#define PRESOL_NAME            "dualagg"
#define PRESOL_DESC            "aggregate variables by dual arguments"
#define PRESOL_PRIORITY            12000     /**< priority of the presolver (>= 0: before, < 0: after constraint handlers) */
#define PRESOL_MAXROUNDS              -1     /**< maximal number of presolving rounds the presolver participates in (-1: no limit) */
#define PRESOL_DELAY               FALSE     /**< should presolver be delayed, if other presolvers found reductions? */

/** type of aggregation */
enum AggrType
{
   BIN0UBOUND = -1,         /**< x_j = u_j + (l_j-u_j)x_i with x_i binary and x_j aggregation variable */
   NOAGG      =  0,         /**< do not aggregate */
   BIN0LBOUND =  1          /**< x_j = l_j + (u_j-l_j)x_i with x_i binary and x_j aggregation variable */
};
typedef enum AggrType AGGRTYPE;

/*
 * Local methods
 */

/** find row which causes the uplock */
static
void getUplockRowIdx(
    SCIPMILPMATRIX*       matrix,             /**< constraint matrix */
    int                   aggvaridx,          /**< index of variable which should be aggregated */
    int*                  rowidx,             /**< row index of uplock */
    SCIP_Real*            coef                /**< coefficient of variable */
   )
{
   int* colpnt;
   int* colend;
   SCIP_Real* valpnt;

   assert(SCIPmatrixGetColNUplocks(matrix, aggvaridx) == 1);

   colpnt = SCIPmatrixGetColIdxPtr(matrix, aggvaridx);
   colend = colpnt + SCIPmatrixGetColNNonzs(matrix, aggvaridx);
   valpnt = SCIPmatrixGetColValPtr(matrix, aggvaridx);

   *rowidx = -1;
   for(; (colpnt < colend); colpnt++, valpnt++)
   {
      /* currently we support only >= relation */
      if( SCIPmatrixIsRowRhsInfinity(matrix, *colpnt) && *valpnt < 0.0 )
      {
         *rowidx = *colpnt;
         *coef = *valpnt;
         break;
      }
   }
}

/** find row which causes the downlock */
static
void getDownlockRowIdx(
    SCIPMILPMATRIX*       matrix,             /**< constraint matrix */
    int                   aggvaridx,          /**< index of variable which should be aggregated */
    int*                  rowidx,             /**< row index of downlock */
    SCIP_Real*            coef                /**< coefficient of variable */
   )
{
   int* colpnt;
   int* colend;
   SCIP_Real* valpnt;

   assert(SCIPmatrixGetColNDownlocks(matrix, aggvaridx) == 1);

   colpnt = SCIPmatrixGetColIdxPtr(matrix, aggvaridx);
   colend = colpnt + SCIPmatrixGetColNNonzs(matrix, aggvaridx);
   valpnt = SCIPmatrixGetColValPtr(matrix, aggvaridx);

   *rowidx = -1;
   for(; (colpnt < colend); colpnt++, valpnt++)
   {
      /* currently we support only >= relation */
      if( SCIPmatrixIsRowRhsInfinity(matrix, *colpnt) && *valpnt > 0.0 )
      {
         *rowidx = *colpnt;
         *coef = *valpnt;
         break;
      }
   }
}

/** find fitting binary variable aggregation for uplock case */
static
void getBinVarIdxInUplockRow(
    SCIP*                 scip,               /**< SCIP main data structure */
    SCIPMILPMATRIX*       matrix,             /**< constraint matrix */
    int                   aggvaridx,          /**< index of variable which should be aggregated */
    int*                  binvaridx,          /**< index of binary variable */
    AGGRTYPE*             aggtype             /**< type of aggregation */
   )
{
   int rowidx;
   SCIP_Real coef;
   int* rowpnt;
   int* rowend;
   SCIP_Real* valpnt;
   SCIP_Real minact;
   SCIP_Real maxact;
   SCIP_Real lhs;
   SCIP_Real lb;
   SCIP_Real ub;

   assert(binvaridx != NULL);
   assert(aggtype != NULL);

   *binvaridx = -1;
   *aggtype = NOAGG;

   getUplockRowIdx(matrix, aggvaridx, &rowidx, &coef);

   if( rowidx < 0 )
      return;

   assert(coef < 0);
   minact = SCIPmatrixGetRowMinActivity(matrix, rowidx);
   maxact = SCIPmatrixGetRowMaxActivity(matrix, rowidx);

   if( SCIPisInfinity(scip, -minact) || SCIPisInfinity(scip, maxact) )
      return;

   lhs = SCIPmatrixGetRowLhs(matrix, rowidx);
   lb = SCIPmatrixGetColLb(matrix, aggvaridx);
   ub = SCIPmatrixGetColUb(matrix, aggvaridx);

   /* search for appropriate binary variables */
   rowpnt = SCIPmatrixGetRowIdxPtr(matrix, rowidx);
   rowend = rowpnt + SCIPmatrixGetRowNNonzs(matrix, rowidx);
   valpnt = SCIPmatrixGetRowValPtr(matrix, rowidx);
   for( ; (rowpnt < rowend); rowpnt++, valpnt++ )
   {
      SCIP_VAR* var;

      if( *rowpnt == aggvaridx )
         continue;

      var = SCIPmatrixGetVar(matrix, *rowpnt);

      if( SCIPvarGetType(var) == SCIP_VARTYPE_BINARY )
      {
         SCIP_Real bincoef;
         bincoef = *valpnt;

         if( bincoef < 0 )
         {
            /* implies binvar = 0, that the constraint is redundant */
            if( SCIPisGE(scip, minact-bincoef, lhs) )
            {
               /* implies binvar = 1, that aggvar = lb */
               SCIP_Real bnd;
               bnd = (lhs - maxact + coef*lb - bincoef) / coef;
               if( SCIPisGE(scip, lb, bnd) )
               {
                  *binvaridx = *rowpnt;
                  *aggtype = BIN0UBOUND;
                  break;
               }
            }
         }

         if( bincoef > 0 )
         {
            /* implies binvar = 1, that the constraint is redundant */
            if( SCIPisGE(scip, minact+bincoef, lhs) )
            {
               /* implies binvar = 0, that aggvar = lb */
               SCIP_Real bnd;
               bnd = (lhs - maxact + coef*lb + bincoef) / coef;
               if( SCIPisGE(scip, lb, bnd) )
               {
                  *binvaridx = *rowpnt;
                  *aggtype = BIN0LBOUND;
               }
            }
         }
      }
   }
}

/** find fitting binary variable aggregation for downlock case */
static
void getBinVarIdxInDownlockRow(
    SCIP*                 scip,               /**< SCIP main data structure */
    SCIPMILPMATRIX*       matrix,             /**< constraint matrix */
    int                   aggvaridx,          /**< index of variable which should be aggregated */
    int*                  binvaridx,          /**< index of binary variable */
    AGGRTYPE*             aggtype             /**< type of aggregation */
   )
{
   int rowidx;
   SCIP_Real coef;
   int* rowpnt;
   int* rowend;
   SCIP_Real* valpnt;
   SCIP_Real minact;
   SCIP_Real maxact;
   SCIP_Real lhs;
   SCIP_Real lb;
   SCIP_Real ub;

   assert(binvaridx != NULL);
   assert(aggtype != NULL);

   *binvaridx = -1;
   *aggtype = NOAGG;

   getDownlockRowIdx(matrix, aggvaridx, &rowidx, &coef);

   if( rowidx < 0 )
      return;

   assert(coef > 0);
   minact = SCIPmatrixGetRowMinActivity(matrix, rowidx);
   maxact = SCIPmatrixGetRowMaxActivity(matrix, rowidx);

   if( SCIPisInfinity(scip, -minact) || SCIPisInfinity(scip, maxact) )
      return;

   lhs = SCIPmatrixGetRowLhs(matrix, rowidx);
   lb = SCIPmatrixGetColLb(matrix, aggvaridx);
   ub = SCIPmatrixGetColUb(matrix, aggvaridx);

   /* search for appropriate binary variables */
   rowpnt = SCIPmatrixGetRowIdxPtr(matrix, rowidx);
   rowend = rowpnt + SCIPmatrixGetRowNNonzs(matrix, rowidx);
   valpnt = SCIPmatrixGetRowValPtr(matrix, rowidx);
   for( ; (rowpnt < rowend); rowpnt++, valpnt++ )
   {
      SCIP_VAR* var;

      if( *rowpnt == aggvaridx )
         continue;

      var = SCIPmatrixGetVar(matrix, *rowpnt);

      if( SCIPvarGetType(var) == SCIP_VARTYPE_BINARY )
      {
         SCIP_Real bincoef;
         bincoef = *valpnt;

         if( bincoef < 0 )
         {
            /* implies binvar = 0, that the constraint is redundant */
            if( SCIPisGE(scip, minact-bincoef, lhs) )
            {
               /* implies binvar = 1, that aggvar = ub */
               SCIP_Real bnd;
               bnd = (lhs - maxact + coef*ub - bincoef) / coef;
               if( SCIPisGE(scip, bnd, ub) )
               {
                  *binvaridx = *rowpnt;
                  *aggtype = BIN0LBOUND;
                  break;
               }
            }
         }

         if( bincoef > 0 )
         {
            /* implies binvar = 1, that the constraint is redundant */
            if( SCIPisGE(scip, minact+bincoef, lhs) )
            {
               /* implies binvar = 0, that aggvar = ub */
               SCIP_Real bnd;
               bnd = (lhs - maxact + coef*ub + bincoef) / coef;
               if( SCIPisGE(scip, bnd, ub) )
               {
                  *binvaridx = *rowpnt;
                  *aggtype = BIN0UBOUND;
                  break;
               }
            }
         }
      }
   }
}

/** find variable aggregations for uplock case */
static
SCIP_RETCODE findUplockAggregations(
   SCIP*                 scip,               /**< SCIP main data structure */
   SCIPMILPMATRIX*       matrix,             /**< constraint matrix */
   int*                  nvaragg,            /**< number of redundant variables */
   AGGRTYPE*             aggtypes,           /**< type of aggregations */
   SCIP_VAR**            aggvars,            /**< pointers to the variables which should by aggregated */
   SCIP_VAR**            binvars             /**< pointers to the binary variables */
   )
{
   int nvars;
   int i;

   assert(scip != NULL);
   assert(matrix != NULL);
   assert(nvaragg != NULL);
   assert(aggtypes != NULL);
   assert(aggvars != NULL);
   assert(binvars != NULL);

   nvars = SCIPmatrixGetNColumns(matrix);

   for( i = 0; i < nvars; i++ )
   {
      if( SCIPmatrixGetColNUplocks(matrix, i) == 1 &&
          SCIPisLE(scip, SCIPvarGetObj(SCIPmatrixGetVar(matrix, i)), 0.0) )
      {
         SCIP_Real lb;
         SCIP_Real ub;

         lb = SCIPmatrixGetColLb(matrix, i);
         ub = SCIPmatrixGetColUb(matrix, i);

         if( !SCIPisInfinity(scip, -lb) && !SCIPisInfinity(scip, ub) )
         {
            int binvaridx;
            AGGRTYPE aggtype;
            getBinVarIdxInUplockRow(scip, matrix, i, &binvaridx, &aggtype);

            if( binvaridx >= 0 )
            {
               aggtypes[i] = aggtype;
               aggvars[i] = SCIPmatrixGetVar(matrix, i);
               binvars[i] = SCIPmatrixGetVar(matrix, binvaridx);
               (*nvaragg)++;
            }
         }
      }
   }

   return SCIP_OKAY;
}

/** find variable aggregations for downlock case */
static
SCIP_RETCODE findDownlockAggregations(
   SCIP*                 scip,               /**< SCIP main data structure */
   SCIPMILPMATRIX*       matrix,             /**< constraint matrix */
   int*                  nvaragg,            /**< number of redundant variables */
   AGGRTYPE*             aggtypes,           /**< type of aggregations */
   SCIP_VAR**            aggvars,            /**< pointers to the variables which should by aggregated */
   SCIP_VAR**            binvars             /**< pointers to the binary variables */
   )
{
   int nvars;
   int i;

   assert(scip != NULL);
   assert(matrix != NULL);
   assert(nvaragg != NULL);
   assert(aggtypes != NULL);
   assert(aggvars != NULL);
   assert(binvars != NULL);

   nvars = SCIPmatrixGetNColumns(matrix);

   for( i = 0; i < nvars; i++ )
   {
      if( SCIPmatrixGetColNDownlocks(matrix, i) == 1 &&
          SCIPisGE(scip, SCIPvarGetObj(SCIPmatrixGetVar(matrix, i)), 0.0) &&
          aggtypes[i] == NOAGG )
      {
         SCIP_Real lb;
         SCIP_Real ub;

         lb = SCIPmatrixGetColLb(matrix, i);
         ub = SCIPmatrixGetColUb(matrix, i);

         if( !SCIPisInfinity(scip, -lb) && !SCIPisInfinity(scip, ub) )
         {
            int binvaridx;
            AGGRTYPE aggtype;
            getBinVarIdxInDownlockRow(scip, matrix, i, &binvaridx, &aggtype);

            if( binvaridx >= 0 )
            {
               aggtypes[i] = aggtype;
               aggvars[i] = SCIPmatrixGetVar(matrix, i);
               binvars[i] = SCIPmatrixGetVar(matrix, binvaridx);
               (*nvaragg)++;
            }
         }
      }
   }

   return SCIP_OKAY;
}

/*
 * Callback methods of presolver
 */


/** execution method of presolver */
static
SCIP_DECL_PRESOLEXEC(presolExecDualagg)
{  /*lint --e{715}*/
   SCIPMILPMATRIX* matrix;
   SCIP_Bool initialized;
   SCIP_Bool complete;

   assert(result != NULL);
   *result = SCIP_DIDNOTRUN;

   if( (SCIPgetStage(scip) != SCIP_STAGE_PRESOLVING) || SCIPinProbing(scip) || SCIPisNLPEnabled(scip) )
      return SCIP_OKAY;

   if( SCIPisStopped(scip) || SCIPgetNActivePricers(scip) > 0 )
      return SCIP_OKAY;

   if( SCIPgetNBinVars(scip) == 0 )
      return SCIP_OKAY;

   *result = SCIP_DIDNOTFIND;

   matrix = NULL;
   SCIP_CALL( SCIPmatrixCreate(scip, &matrix, &initialized, &complete) );

   if( initialized && complete )
   {
      int nvaragg;
      AGGRTYPE* aggtypes;
      SCIP_VAR** aggvars;
      SCIP_VAR** binvars;
      int ncols;
      int nrows;

      ncols = SCIPmatrixGetNColumns(matrix);
      nrows = SCIPmatrixGetNRows(matrix);
      nvaragg = 0;

      SCIP_CALL( SCIPallocBufferArray(scip, &aggtypes, ncols) );
      BMSclearMemoryArray(aggtypes, ncols);

      SCIP_CALL( SCIPallocBufferArray(scip, &aggvars, ncols) );
      SCIP_CALL( SCIPallocBufferArray(scip, &binvars, ncols) );

      SCIP_CALL( findUplockAggregations(scip, matrix, &nvaragg, aggtypes, aggvars, binvars) );
      SCIP_CALL( findDownlockAggregations(scip, matrix, &nvaragg, aggtypes, aggvars, binvars) );

      if( nvaragg > 0 )
      {
         int v;
         for( v = 0; v < ncols; v++ )
         {
            if( aggtypes[v] != NOAGG )
            {
               SCIP_Bool infeasible;
               SCIP_Bool redundant;
               SCIP_Bool aggregated;
               SCIP_Real ub;
               SCIP_Real lb;

               ub = SCIPmatrixGetColUb(matrix, v);
               lb = SCIPmatrixGetColLb(matrix, v);

               /* aggregate variable */
               assert(aggvars[v] != NULL && binvars[v] != NULL);
               if( aggtypes[v] == BIN0UBOUND )
               {
                  SCIP_CALL( SCIPaggregateVars(scip, aggvars[v], binvars[v], 1.0, ub-lb,
                        ub, &infeasible, &redundant, &aggregated) );
               }
               else
               {
                  SCIP_CALL( SCIPaggregateVars(scip, aggvars[v], binvars[v], 1.0, lb-ub,
                        lb, &infeasible, &redundant, &aggregated) );
               }
               if( infeasible )
               {
                  SCIPdebugMessage(" -> infeasible aggregation\n");
                  *result = SCIP_CUTOFF;
                  return SCIP_OKAY;
               }

               if( aggregated )
               {
                  (*naggrvars)++;

                  /* set result pointer */
                  if( (*result) == SCIP_DIDNOTFIND )
                     *result = SCIP_SUCCESS;
               }
            }
         }
      }

      SCIPfreeBufferArray(scip, &binvars);
      SCIPfreeBufferArray(scip, &aggvars);
      SCIPfreeBufferArray(scip, &aggtypes);
   }

   SCIPmatrixFree(scip, &matrix);

   return SCIP_OKAY;
}

/*
 * presolver specific interface methods
 */

/** creates the dualagg presolver and includes it in SCIP */
SCIP_RETCODE SCIPincludePresolDualagg(
   SCIP*                 scip                /**< SCIP data structure */
   )
{
   SCIP_PRESOL* presol;

   /* include presolver */
   SCIP_CALL( SCIPincludePresolBasic(scip, &presol, PRESOL_NAME, PRESOL_DESC, PRESOL_PRIORITY, PRESOL_MAXROUNDS,
         PRESOL_DELAY, presolExecDualagg, NULL) );

   return SCIP_OKAY;
}
